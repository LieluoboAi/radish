package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utfcpp",
    srcs = [
        "source/utf8.h",
    ] + glob([
        "source/*/*.h",
    ]),
    hdrs = [
        "source/utf8.h",
    ],
    includes = [
        ".",
        "source",
    ],
    visibility = ["//visibility:public"],
)
