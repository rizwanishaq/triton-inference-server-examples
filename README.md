---
title: "Triton Inference Server Series: Streaming with Python Backend"
description: Learn how to use the Python backend of NVIDIA Triton Inference Server to perform streaming inference.
image: "../../public/blogs/1_nautKVzRYdwRmIUtWJmlLw.png"
publishedAt: "2023-07-09"
updatedAt: "2023-07-09"
author: "Rizwan Ishaq"
isPublished: true
tags:
  - Python
  - Triton Inference Servering
  - Streaming
  - Machine learning
---

# Introduction

Tired of the same old batch inference? Ready to take your Python skills to the next level with streaming inference? Then look no further than NVIDIA Triton Inference Server!

With Triton, you can use your Python skills to serve machine learning models in real time, right from your GPU or CPU. And with its streaming inference capabilities, you can process data as it's being generated, without having to wait for batches to accumulate.

In this blog post, I'll show you how to use the Python backend of Triton to perform streaming inference on images. I'll explain each step in detail, so you'll be able to get started even if you're new to Triton.

## What is Triton Inference Server

According to [triton-inference-server](https://github.com/triton-inference-server/server)

> " **Triton Inference Server** is an open source inference serving software that streamlines AI inferencing. Triton enables teams to deploy any AI model from multiple deep learning and machine learning frameworks, including TensorRT, TensorFlow, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more. Triton Inference Server supports inference across cloud, data center, edge and embedded devices on NVIDIA GPUs, x86 and ARM CPU, or AWS Inferentia. Triton Inference Server delivers optimized performance for many query types, including real time, batched, ensembles and audio/video streaming. Triton inference Server is part of NVIDIA AI Enterprise, a software platform that accelerates the data science pipeline and streamlines the development and deployment of production AI.
> "

## Project Setup

Before we dive into code, let's grasp the essence of Memcached. According to the [official website](https://www.memcached.org/):

> "**Free & open source, high-performance, distributed memory object caching system**, generic in nature, but intended for use in speeding up dynamic web applications by alleviating database load. Memcached is an in-memory key-value store for small chunks of arbitrary data (strings, objects) from results of database calls, API calls, or page rendering."

To gain a hands-on understanding, I recommend watching [this video](https://www.youtube.com/watch?v=NCePGsRZFus) which offers a practical insight into Memcached's capabilities.

## Setting Up Memcached with Docker

To kickstart our journey, we need to set up a Memcached server. Docker provides an effortless solution for this. Assuming Docker is installed on your machine (if not, you can install it [here](https://www.docker.com/get-started)), follow these steps:

```bash
$ docker run --name memcachedserver -p 11211:11211 -d memcached
```

This command launches the Memcached server, making it accessible on port 11211. Verify if the server is running using the following command:

```bash
$ docker ps
```

With the Memcached server up and running, we're ready to interact with it using Node.js.

## Implementing Memcached in Node.js

Let's create a directory named MemcachedCrashCourse for our project:

```bash
$ mkdir MamcachedCrashCourse
$ cd MamcachedCrashCourse
```

Next, initialize the project and install the necessary packages:

```bash
$ npm init -y
$ npm install memcached
```

Now, create a new file named index.js:

```bash
$ touch index.js
```

Add the following code to index.js:

```javascript
/**
 * This script demonstrates how to connect to a Memcached server,
 * perform a write operation, and subsequently read the data.
 */

const os = require("os");
const MEMCACHED = require("memcached");
const serverPool = new MEMCACHED([`${os.hostname()}:11211`]);

/**
 * Writes the value "bar" under the key "foo" to the Memcached server.
 */
const write = () => {
  serverPool.set("foo", "bar", 3600, (error) => {
    if (error) {
      console.log(error);
    }
  });
};

/**
 * Reads the value associated with the key "foo" from the Memcached server.
 */
const read = () => {
  serverPool.get("foo", (error, data) => {
    if (error) {
      console.log(error);
    }
    console.log(data);
  });
};

// Execute the write operation
write();

// Execute the read operation
read();
```

In this code, we first import the necessary packages (`os` and `memcached`) and connect to the Memcached server using the hostname and port. The `write` function stores the value `"bar"` under the key `"foo"` in the Memcached server. Subsequently, the read function retrieves the value associated with the key `"foo"`. When you run the `index.js` file with the command `node index.js`, you should see the output `bar`.

This simple example demonstrates how to integrate Memcached into a Node.js application without the need for additional configuration. By leveraging the power of Memcached, you can significantly enhance the performance of your dynamic web applications.
