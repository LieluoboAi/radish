package(
    default_visibility = ["//visibility:public"],
)

proto_library(
  name="example_proto",
  srcs=[
    "example.proto",
  ]
)

cc_proto_library(
  name="example_proto_cc",
  deps=[
    ":example_proto",
  ]
)
