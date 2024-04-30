#pragma once

#include "yolov8.hpp"
#include "classification_model.hpp"

const Yolov8Params<ClassificationModel::Detection> MOHNISH7{
  { "images" },
  { 1, 3, 640, 640 },
  { 1, 7, 8400 },
  { "output0" },
  {
    ClassificationModel::Detection::ARROW_RIGHT,
    ClassificationModel::Detection::ARROW_LEFT,
    ClassificationModel::Detection::CONE,
  },
  ClassificationModel::Detection::NONE,
};

using Yolov8ArrowClassifier = Yolov8Detector<ClassificationModel::Detection>;
static Yolov8ArrowClassifier make_mohnish7_model(
  const std::string& model_path)
{
  return Yolov8ArrowClassifier(model_path, MOHNISH7);
}
