/*
 * File: logging.h
 * Project: utils
 * Author: koth (Koth Chen)
 * -----
 * Last Modified: 2019-09-24 10:06:23
 * Modified By: koth (nobody@verycool.com)
 * -----
 * Copyright 2020 - 2019
 */
#pragma once
#include "glog/logging.h"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"


// #define SPDLOG_TRACE(logger, ...) if (logger->should_log(level::trace)){logger->trace("{}::{}()#{}: ", __FILE__ , __FUNCTION__, __LINE__, fmt::format(__VA_ARGS__));}