package(default_visibility = ["//visibility:public"])

load("//google/protobuf:protobuf.bzl", "cc_proto_library")
load("//tensorflow/core:platform/default/genproto.bzl", "proto_java_library")

cc_binary(
    name = "libtaggerflow.so",
    srcs = ["taggerflow.cc",
            "taggerflow.h"],
    linkshared=1,
    deps = [
        ":cc_protos",
        "@jni//:headers",
        "//tensorflow/core:tensorflow",
    ],
)

cc_proto_library(
    name = "cc_protos",
    srcs = ["tagging.proto"],
)

proto_java_library(
    name = "java_protos",
    src = "tagging.proto"
)

java_binary(
    name = "taggerflow",
    srcs = glob(["**/*.java"]),
    main_class = "edu.uw.Taggerflow",
    deps = [
        ":java_protos",
        "libtaggerflow.so",
        "@bazel_tools//third_party:protobuf",
    ]
)
