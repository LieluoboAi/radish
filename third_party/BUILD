package(
    default_visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pytorch",
    deps = select({
        ":darwin": [
            "@libtorch_mac//:libtorch",
        ],
        "//conditions:default": [
            "@libtorch_unix//:libtorch",
        ],
    }),
)
