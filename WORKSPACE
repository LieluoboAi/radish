workspace(name = "com_lieluobo_radish")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

load("//:bazel/repository.bzl", "radish_repositories")

radish_repositories()

bind(
    name = "nanopb",
    actual = "@com_github_nanopb_nanopb//:nanopb",
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


bind(
    name = "protocol_compiler",
    actual = "@com_google_protobuf//:protoc",
)



bind(
    name = "protobuf_headers",
    actual = "@com_google_protobuf//:protobuf_headers",
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()


load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()


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


load("@com_github_lieluoboai_leveldb//:bazel/repositories.bzl", "repositories")

repositories()

load("@com_github_lieluoboai_snappy//:bazel/repositories.bzl", "repositories")

repositories()

load("@com_github_lieluoboai_crc32c//:bazel/repositories.bzl", "repositories")

repositories()


bind(
    name="absl_string",
    actual="@com_google_absl//absl/strings",
)