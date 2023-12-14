const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");
const util = require("util");
const PROTO_PATH = __dirname + "/../../../protos/grpc_service.proto";
const PROTO_IMPORT_PATH = __dirname + "/../../../protos";

const fs = require("fs");
const wav = require("wav");

export default {
  init,
  run,
};

let client: any;

async function init(params: any) {
  const { target, log } = params;

  const GRPCServicePackageDefinition = protoLoader.loadSync(PROTO_PATH, {
    includeDirs: [PROTO_IMPORT_PATH],
    keepCase: true,
    longs: String,
    enums: String,
    defaults: true,
    oneofs: true,
  });

  const inference = grpc.loadPackageDefinition(
    GRPCServicePackageDefinition
  ).inference;

  const isSecured = target.includes(":443");
  const credentials = isSecured
    ? grpc.credentials.createSsl()
    : grpc.credentials.createInsecure();
  client = new inference.GRPCInferenceService(target, credentials);

  log.info({}, `XTTS target ${target} (secured:${isSecured})`);
}

async function run(params: any): Promise<Buffer> {
  const { language, text, uuid, log } = params;

  const shortLang: string = getShortLangCode(language);

  log.info({}, `[${uuid}] XTTS: (${language}) "${text}"`);

  /////////////////////

  const speaker = "papanoel"; // speaker name, for now we only suport 'papanoel'

  /////////////////////

  async function main(): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      const argv = process.argv.slice(2);
      const model_version = "";
      const batch_size = 1;
      const dimension = 16;

      const model_name = "xtts";

      const deadline = Date.now() + 240 * 1000;

      const input_dict = {
        language: shortLang,
        speaker: speaker,
        text: text,
      };

      const input = JSON.stringify(input_dict);

      log.info({}, `[${uuid}] XTTS input: ${input}`);

      const requestId = uuid; // 'uid_' + Math.round(100000000 * Math.random());
      const metadata = new grpc.Metadata();
      metadata.add("request_id", requestId);

      const modelInferRequest = {
        model_name: model_name,
        id: requestId,
        inputs: [{ name: "INPUT_0", datatype: "BYTES", shape: [1] }],
        outputs: [{ name: "OUTPUT_0" }],
        raw_input_contents: [stringToBuffer(input)],
        parameters: {
          request_id: { int64_param: requestId },
          triton_enable_empty_final_response: { bool_param: true },
        },
      };

      let chunkCounter = 0;

      const call = client.ModelStreamInfer(metadata, { deadline });

      call.write(modelInferRequest);

      const blocks: any[] = [];

      call.on("data", (data: any) => {
        const isFinal =
          data.infer_response.parameters.triton_final_response.bool_param;

        log.info({}, `[${uuid}] XTTS data (isFinal:${isFinal})`);

        if (isFinal === true) {
          log.info({}, `[${uuid}] XTTS END`);
          const buf: Buffer = Buffer.concat(blocks);
          const downsampledAudio = downsample(buf);
          // fileWriter.write(int16data);
          call.end();
          resolve(downsampledAudio);
        } else {
          const raw = data.infer_response.raw_output_contents[0];
          if (raw && raw.length > 0) {
            const audio_content = Buffer.from(raw);
            blocks.push(audio_content);
          }
        }

        chunkCounter++;
      });

      call.on("error", (error: any) => {
        log.error(
          { error },
          `[${uuid}] XTTS ERROR ${error.details} (${
            error.code
          }) ${JSON.stringify(error)}`
        );
        call.end();
        reject(error);
      });

      call.on("end", () => {
        // console.log('END =======================');
      });
    });
  }

  return main();
}

function downsample(buf: Buffer) {
  const outputBuffer = Buffer.alloc(buf.length / 3, 0);

  for (let i = 0; i < buf.length; i += 2 * 3) {
    if (i + 6 > buf.length) {
      break;
    }
    const [data1, data2, data3] = [
      buf.readInt16LE(i),
      buf.readInt16LE(i + 2),
      buf.readInt16LE(i + 4),
    ];
    const data = (data1 + data2 + data3) / 3;

    if (i / 3 < outputBuffer.length - 1) {
      outputBuffer.writeInt16LE(data, i / 3);
    }
  }

  return outputBuffer;
}

function stringToBuffer(str: string) {
  // Allocate a 4-byte buffer to store the length of the string
  const lengthBuffer = Buffer.alloc(4);

  // Write the length of the string to the buffer
  lengthBuffer.writeUInt32LE(Buffer.from(str, "utf8").length, 0);

  // Concatenate the length buffer with the buffer from the string
  return Buffer.concat([lengthBuffer, Buffer.from(str, "utf8")]);
}

function getShortLangCode(longCode: string): string {
  if (longCode.length === 2) {
    return longCode;
  }

  const shortCodes: any = {
    "en-US": "en",
    "en-GB": "en",
    "es-ES": "es",
    "fr-FR": "fr",
    "de-DE": "de",
    "it-IT": "it",
    "pt-PT": "pt",
    "pl-PL": "pl",
    "tr-TR": "tr",
    "ru-RU": "ru",
    nl: "nl",
    "nl-NL": "nl",
    "nl-BE": "nl",
    "cs-CZ": "cs",
    "ar-AE": "ar",
    "zh-cn": "zh-cn",
    "hu-HU": "hu",
    "ko-KR": "ko",
  };

  return shortCodes[longCode];
}
