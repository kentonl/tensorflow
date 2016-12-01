package(default_visibility = ["//visibility:public"])

load("@io_bazel//tools/build_rules:genproto.bzl", "java_proto_library")
load("@protobuf//:protobuf.bzl", "cc_proto_library")

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
    protoc = "@protobuf//:protoc",
    default_runtime = "@protobuf//:protobuf",
)

java_proto_library(
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
        "@io_bazel//third_party/protobuf/3.0.0:protobuf",
    ]
)
