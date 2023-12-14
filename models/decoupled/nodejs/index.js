// Import required modules
const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");
const utils = require("./utils/utils");

/**
 * Path to the gRPC service protobuf definition file.
 * @type {string}
 */
const PROTO_PATH = __dirname + "/protos/grpc_service.proto";

/**
 * Path to the directory containing imported protobuf files.
 * @type {string}
 */
const PROTO_IMPORT_PATH = __dirname + "/protos";

/**
 * Converts a string to a Buffer using utility function.
 * @type {Function}
 */
const stringToBuffer = utils.stringToBuffer;

// Define constants for the model and service URL
const model_name = "square_int32";
const url = "0.0.0.0:8001";

// Load gRPC service definition
const GRPCServicePackageDefinition = protoLoader.loadSync(PROTO_PATH, {
  includeDirs: [PROTO_IMPORT_PATH],
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});

// Extract the 'inference' package from the loaded definition
const inference = grpc.loadPackageDefinition(
  GRPCServicePackageDefinition
).inference;

// Create a gRPC client for the Inference Service
const client = new inference.GRPCInferenceService(
  url,
  grpc.credentials.createInsecure()
);

// Define the model inference request
const modelInferRequest = {
  model_name: model_name,
  inputs: [{ name: "IN", datatype: "BYTES", shape: [1] }],
  outputs: [{ name: "OUT" }],
  raw_input_contents: [stringToBuffer("hi how are you?")],
  parameters: {
    triton_enable_empty_final_response: { bool_param: true },
  },
};

// Initialize metadata for the gRPC call
const metadata = new grpc.Metadata();

// Make a model stream inference gRPC call
const call = client.ModelStreamInfer(metadata, {});

// Write the model inference request to the gRPC call
call.write(modelInferRequest);

// Handle incoming data from the gRPC call
call.on("data", (data) => {
  // Check if it's the final response
  const end_flag =
    data.infer_response.parameters["triton_final_response"].bool_param;

  // Log data if not the final response
  if (!end_flag) {
    console.log(
      `data: ${data.infer_response.raw_output_contents[0].toString(
        "utf8",
        4
      )} - end_flag: ${end_flag}`
    );
  }

  // End the gRPC call if it's the final response
  if (end_flag) {
    call.end();
  }
});

// Handle errors during the gRPC call
call.on("error", (error) => {
  console.log("ERROR", error);
});

// Handle the end of the gRPC call
call.on("end", () => {
  console.log("END");
});
