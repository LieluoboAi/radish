workspace(name = "com_koth_knlp")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

http_archive(
    name = "com_googlesource_code_re2",
    sha256 = "e57eeb837ac40b5be37b2c6197438766e73343ffb32368efea793dfd8b28653b",
    strip_prefix = "re2-26cd968b735e227361c9703683266f01e5df7857",
    urls = [
        "https://mirror.bazel.build/github.com/google/re2/archive/26cd968b735e227361c9703683266f01e5df7857.tar.gz",
        "https://github.com/google/re2/archive/26cd968b735e227361c9703683266f01e5df7857.tar.gz",
    ],
)

# C++ rules for Bazel.
http_archive(
    name = "rules_cc",
    sha256 = "67412176974bfce3f4cf8bdaff39784a72ed709fc58def599d1f68710b58d68b",
    strip_prefix = "rules_cc-b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e",
    urls = [
        "https://github.com/bazelbuild/rules_cc/archive/b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e.zip",
    ],
)

http_archive(
    name = "boringssl",
    sha256 = "524ba98a56300149696481b4cb9ddebd0c7b7ac9b9f6edee81da2d2d7e5d2bb3",
    strip_prefix = "boringssl-a0fb951d2a26a8ee746b52f3ba81ab011a0af778",
    urls = [
        "https://mirror.bazel.build/github.com/google/boringssl/archive/a0fb951d2a26a8ee746b52f3ba81ab011a0af778.tar.gz",
        "https://github.com/google/boringssl/archive/a0fb951d2a26a8ee746b52f3ba81ab011a0af778.tar.gz",
    ],
)

http_archive(
    name = "com_github_nanopb_nanopb",
    build_file = "@com_github_grpc_grpc//third_party:nanopb.BUILD",
    sha256 = "8bbbb1e78d4ddb0a1919276924ab10d11b631df48b657d960e0c795a25515735",
    strip_prefix = "nanopb-f8ac463766281625ad710900479130c7fcb4d63b",
    urls = [
        # "https://storage.googleapis.com/mirror.tensorflow.org/github.com/nanopb/nanopb/archive/f8ac463766281625ad710900479130c7fcb4d63b.tar.gz",
        "https://github.com/nanopb/nanopb/archive/f8ac463766281625ad710900479130c7fcb4d63b.tar.gz",
    ],
)

bind(
    name = "nanopb",
    actual = "@com_github_nanopb_nanopb//:nanopb",
)

http_archive(
    name = "zlib_archive",
    build_file = "//third_party:zlib.BUILD",
    sha256 = "36658cb768a54c1d4dec43c3116c27ed893e88b02ecfcb44f2166f9c0b7f2a0d",
    strip_prefix = "zlib-1.2.8",
    urls = [
        # "http://bazel-mirror.storage.googleapis.com/zlib.net/zlib-1.2.8.tar.gz",
        "http://zlib.net/fossils/zlib-1.2.8.tar.gz",
    ],
)

http_archive(
    name = "boost",
    build_file = "//third_party:boost.BUILD",
    patch_tool = "patch",
    patches = ["//third_party:boost1.68.patch"],
    sha256 = "7f6130bc3cf65f56a618888ce9d5ea704fa10b462be126ad053e80e553d6d8b7",
    strip_prefix = "boost_1_68_0/",
    type = "tar.bz2",
    urls = [
        "https://sourceforge.net/projects/boost/files/boost/1.68.0/boost_1_68_0.tar.bz2",
    ],
)

http_archive(
    name = "sentencepiece",
    sha256 = "4567a038605b7da094089ae1dfe7ef7ed623c7aa76050a74aeaf7fd84b3fe390",
    strip_prefix = "sentencepiece-998ab31fe1388fd3b729079138d65f89b9a5c62e",
    urls = [
        "https://github.com/lieluoboai/sentencepiece/archive/998ab31fe1388fd3b729079138d65f89b9a5c62e.tar.gz",
    ],
)

http_archive(
    name = "com_google_protobuf",
    sha256 = "f1748989842b46fa208b2a6e4e2785133cfcc3e4d43c17fecb023733f0f5443f",
    strip_prefix = "protobuf-3.7.1",
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/v3.7.1.tar.gz",
    ],
)

http_archive(
    name = "com_google_protobuf_cc",
    sha256 = "f1748989842b46fa208b2a6e4e2785133cfcc3e4d43c17fecb023733f0f5443f",
    strip_prefix = "protobuf-3.7.1",
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/v3.7.1.tar.gz",
    ],
)

bind(
    name = "protocol_compiler",
    actual = "@com_google_protobuf//:protoc",
)

# bind(
#     name = "protobuf_clib",
#     actual = "@com_google_protobuf//:protobuf",
# )

bind(
    name = "protobuf_headers",
    actual = "@com_google_protobuf//:protobuf_headers",
)

http_archive(
    name = "com_google_absl",
    build_file = "//third_party:absl.BUILD",
    sha256 = "7ce0245f27f0446dab92b849492bba85263daacaf9f34ce2fbb2831f43fd04db",
    strip_prefix = "abseil-cpp-502efe6d7841bff82b1aeef5500491fe9a48c635",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/502efe6d7841bff82b1aeef5500491fe9a48c635.tar.gz",
    ],
)

http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "11ac793c562143d52fd440f6549588712badc79211cdc8c509b183cb69bddad8",
    strip_prefix = "grpc-1.22.0",
    urls = [
        # "https://storage.googleapis.com/mirror.tensorflow.org/github.com/grpc/grpc/archive/4566c2a29ebec0835643b972eb99f4306c4234a3.tar.gz",
        "https://github.com/grpc/grpc/archive/v1.22.0.tar.gz",
    ],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

http_archive(
    name = "libtorch_unix",
    build_file = "//third_party:libtorch_unix.BUILD",
    sha256 = "6b0cc8840e05e5e2742e5c59d75f8379f4eda8737aeb24b5ec653735315102b2",
    strip_prefix = "libtorch",
    url = "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.2.0.zip",
)

http_archive(
    name = "libtorch_mac",
    build_file = "//third_party:libtorch_mac.BUILD",
    strip_prefix = "libtorch",
    url = "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.2.0.zip",
)

http_archive(
    name = "curl",
    build_file = "//third_party:curl.BUILD",
    sha256 = "e9c37986337743f37fd14fe8737f246e97aec94b39d1b71e8a5973f72a9fc4f5",
    strip_prefix = "curl-7.60.0",
    urls = [
        "https://mirror.bazel.build/curl.haxx.se/download/curl-7.60.0.tar.gz",
        "https://curl.haxx.se/download/curl-7.60.0.tar.gz",
    ],
)

http_archive(
    name = "jsoncpp",
    build_file = "//third_party:jsonpp.BUILD",
    sha256 = "07d34db40593d257324ec5fb9debc4dc33f29f8fb44e33a2eeb35503e61d0fe2",
    strip_prefix = "jsoncpp-11086dd6a7eba04289944367ca82cea71299ed70",
    urls = [
        "https://github.com/open-source-parsers/jsoncpp/archive/11086dd6a7eba04289944367ca82cea71299ed70.tar.gz",
    ],
)

http_archive(
    name = "rules_proto",
    sha256 = "602e7161d9195e50246177e7c55b2f39950a9cf7366f74ed5f22fd45750cd208",
    strip_prefix = "rules_proto-97d8af4dc474595af3900dd85cb3a29ad28cc313",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
        "https://github.com/bazelbuild/rules_proto/archive/97d8af4dc474595af3900dd85cb3a29ad28cc313.tar.gz",
    ],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()

skylib_version = "0.8.0"

http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    type = "tar.gz",
    url = "https://github.com/bazelbuild/bazel-skylib/releases/download/{}/bazel-skylib.{}.tar.gz".format(skylib_version, skylib_version),
)

# Needed by gRPC
bind(
    name = "protobuf",
    actual = "@com_google_protobuf//:protobuf",
)

# gRPC expects //external:protobuf_clib and //external:protobuf_compiler
# to point to Protobuf's compiler library.
bind(
    name = "protobuf_clib",
    actual = "@com_google_protobuf//:protoc_lib",
)

# Needed by gRPC
bind(
    name = "protobuf_headers",
    actual = "@com_google_protobuf//:protobuf_headers",
)

# ===== gRPC dependencies =====
bind(
    name = "libssl",
    actual = "@boringssl//:ssl",
)

bind(
    name = "zlib",
    actual = "@zlib_archive//:zlib",
)

bind(
    name = "madler_zlib",
    actual = "@zlib_archive//:zlib",
)

# gRPC wants the existence of a cares dependence but its contents are not
# actually important since we have set GRPC_ARES=0 in tools/bazel.rc
bind(
    name = "cares",
    actual = "@com_github_grpc_grpc//third_party/nanopb:nanopb",
)

bind(
    name = "grpc_cpp_plugin",
    actual = "@com_github_grpc_grpc//:grpc_cpp_plugin",
)

bind(
    name = "grpc_lib",
    actual = "@com_github_grpc_grpc//:grpc++",
)

bind(
    name = "grpc_lib_unsecure",
    actual = "@com_github_grpc_grpc//:grpc++_unsecure",
)

http_archive(
    name = "spdlog",
    build_file = "//third_party:spdlog.BUILD",
    strip_prefix = "spdlog-a7148b718ea2fabb8387cb90aee9bf448da63e65",
    sha256 = "c85af153ad14cf26e32396ba4d19d2371b16f99151f23aa3189e99e161813cdb",
    urls = [
        "https://github.com/gabime/spdlog/archive/a7148b718ea2fabb8387cb90aee9bf448da63e65.tar.gz",
    ],
)

http_archive(
    name = "fmtlib",
    build_file = "//third_party:fmtlib.BUILD",
    strip_prefix = "fmt-9e554999ce02cf86fcdfe74fe740c4fe3f5a56d5",
    sha256 = "f131a0ecef42c84f59c087f9a1ab06cedb38c66e35b1cee231056b5e888e8cf6",
    urls = [
        "https://github.com/fmtlib/fmt/archive/9e554999ce02cf86fcdfe74fe740c4fe3f5a56d5.tar.gz",
    ],
)

http_archive(
    name = "com_github_lieluoboai_leveldb",
    sha256 = "2c8815db8f1b5031e62d530e13ef31242f85ebcc6c7b486d8897474df482786f",
    strip_prefix = "leveldb-3d51bafc1764d7115db5f83b4a838bc6e630449a",
    urls = [
        "https://github.com/lieluoboai/leveldb/archive/3d51bafc1764d7115db5f83b4a838bc6e630449a.tar.gz",
    ],
)

http_archive(
    name = "gulrak_filesystem",
    build_file = "//third_party:gulrak_filesystem.BUILD",
    strip_prefix = "filesystem-15a82e9098bcd30125248dc84dc95fc0402f3b40",
    sha256 = "5cfcd5c28600543b7f718f7661c4f892737c3c0ff1803ba9ac052ba2688f94a3",
    urls = [
        "https://github.com/gulrak/filesystem/archive/15a82e9098bcd30125248dc84dc95fc0402f3b40.tar.gz",
    ],
)


load("@com_github_lieluoboai_leveldb//:bazel/repositories.bzl", "repositories")

repositories()

load("@com_github_lieluoboai_snappy//:bazel/repositories.bzl", "repositories")

repositories()

load("@com_github_lieluoboai_crc32c//:bazel/repositories.bzl", "repositories")

repositories()
