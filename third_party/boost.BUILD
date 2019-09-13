# Description:
#   The Boost library collection (http://www.boost.org)
#
# Most Boost libraries are header-only, in which case you only need to depend
# on :boost. If you need one of the libraries that has a separately-compiled
# implementation, depend on the appropriate libs rule.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Boost software license

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)


cc_library(
    name = "boost",
    hdrs = glob([
        "boost/**/*.hpp",
        "boost/**/*.h",
        "boost/**/*.ipp",
    ],
    exclude = ["**/test/**"]),
    includes = [
    "."
    ],
)

cc_library(
    name = "filesystem",
    srcs = glob([ "libs/filesystem/src/*.cpp","libs/filesystem/src/*.hpp"]),
    deps = [
        ":boost",
        ":system",
    ],
)

cc_library(
    name = "iostreams",
    srcs = glob(["libs/iostreams/src/*.cpp","libs/iostreams/src/*.hpp"]),
    deps = [
        ":boost",
        "@bzip2_archive//:bz2lib",
        "@zlib_archive//:zlib",
    ],
)

cc_library(
    name = "program_options",
    srcs = glob([ "libs/program_options/src/*.cpp","libs/program_options/src/*.hpp"]),
    deps = [
        ":boost",
    ],
)


cc_library(
    name = "thread",
    srcs = glob([ "libs/thread/src/*.cpp","libs/thread/src/*.hpp","libs/thread/src/pthread/*.cpp","libs/thread/src/pthread/*.hpp"]),
    deps = [
        ":boost",
    ],
)


cc_library(
    name = "date_time",
    srcs = glob([ "libs/date_time/src/*/*.cpp","libs/date_time/src/*/*.hpp"]),
    deps = [
        ":boost",
    ],
)

cc_library(
    name = "algorithm",
    srcs = glob([ "libs/regex/src/*.cpp","libs/regex/src/*.hpp"]),
    deps = [
        ":boost",
    ],
)

cc_library(
    name = "system",
    srcs = glob(["libs/system/src/*.cpp","libs/system/src/*.hpp"]),
    defines=select({
        ":darwin": [
        ],
        "//conditions:default": [
#"BOOST_SYSTEM_NO_DEPRECATED","BOOST_ERROR_CODE_HEADER_ONLY"
        ],
    }) +[],
    deps = [
        ":boost",
    ],
)
