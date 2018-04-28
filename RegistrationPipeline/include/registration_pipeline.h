#ifndef REGISTRATION_PIPELINE_H
#define REGISTRATION_PIPELINE_H
#include "prereq.h"

#include <vector>
#include <string>

namespace RegPipeline
{
  // Trim two points clouds
  REGPIPELINE_PUBLIC void TrimPointsClouds(const std::vector<std::string> &filenames, std::string &in_format);
}

#endif