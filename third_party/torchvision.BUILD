package(default_visibility = ["//visibility:public"])


cc_library(
    name = "models",
    hdrs = glob(["torchvision/csrc/models/*.h"]),
    includes = [
        ".",
    ],
    srcs = glob(["torchvision/csrc/models/*.cpp"]),
    deps=[
        "@com_lieluobo_radish//third_party:pytorch",
    ]
    
)
