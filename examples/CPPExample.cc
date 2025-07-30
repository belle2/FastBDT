/**
 * Thomas Keck 2017
 */

#include "Classifier.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>


class IrisData {
public:

  IrisData(const std::string& filename, int nFeatures = 4, const char delimiter = ',')
  {
    std::ifstream file(filename);
    if (!file.is_open())
      throw std::runtime_error("Cannot open file: " + filename);

    std::string line;
    std::vector<std::vector<float>> rows;

    while (std::getline(file, line)) {
      const std::vector<std::string> tokens = split(line, delimiter);
      if (static_cast<int>(tokens.size()) < nFeatures + 1)  // +1 for the label
        throw std::runtime_error("Malformed line: " + line);

      std::vector<float> features(nFeatures);
      for (int i = 0; i < nFeatures; ++i) {
        features[i] = std::stof(tokens[i]);
      }

      rows.push_back(features);
      _labels.push_back(std::stoi(tokens[nFeatures]));
      _weights.push_back(1.0f);
    }

    _X.resize(nFeatures);
    for (int i = 0; i < nFeatures; ++i) {
      _X[i].resize(rows.size());
      for (size_t j = 0; j < rows.size(); ++j) {
        _X[i][j] = rows[j][i];
      }
    }
  }

  const std::vector<std::vector<float>>& getX() const
  {
    return _X;
  }

  std::vector<bool> getY(int targetLabel) const
  {
    std::vector<bool> y;
    for (int label : _labels)
      y.push_back(label == targetLabel);
    return y;
  }

  const std::vector<float>& getW() const
  {
    return _weights;
  }

private:

  static std::vector<std::string> split(const std::string& line, const char delimiter)
  {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
      if (!token.empty())
        tokens.push_back(token);
    }
    return tokens;
  }

  std::vector<std::vector<float>> _X;
  std::vector<int> _labels;
  std::vector<float> _weights;
};


int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <input_file>\n";
    return 1;
  }

  const IrisData data{std::string{argv[1]}};
  const auto X = data.getX();
  const auto y = data.getY(1);
  const auto w = data.getW();

  FastBDT::Classifier classifier;
  // Most of the parameters have default values and
  // you don't have to set them.
  classifier.SetBinning({5, 5, 5, 5}); // 2^5 Bins for each feature, default is 2^8 bins per feature
  classifier.SetNTrees(10); // default is 100
  classifier.SetDepth(3); // default is 3
  classifier.SetShrinkage(0.1); // default is 0.1
  classifier.SetSubsample(0.5); // default is 0.5
  classifier.SetSPlot(false); // default is false
  classifier.SetPurityTransformation({false, false, false, false}); // Do not use purity transformation for the feature, default is false as well
  classifier.SetNumberOfFlatnessFeatures(0); // We do not use uniform boosting here (default is 0 as well)
  classifier.SetFlatnessLoss(-1); // We do not use uniform boosting here (default is -1 as well)
  classifier.SetTransform2Probability(true); // Transform output to probability (default is true)

  classifier.fit(X, y, w);

  auto getIrisScore = [](const FastBDT::Classifier & classifier,
                         const std::vector<std::vector<float>>& X,
			 const std::vector<bool>& y) -> float {
    float sum = 0;
    for (unsigned int i = 0; i < y.size(); ++i)
    {
      const float p = classifier.predict({X[0][i], X[1][i], X[2][i], X[3][i]});
      sum += (static_cast<int>(y[i]) - p) * (static_cast<int>(y[i]) - p);
    }
    return sum;
  };

  std::cout << "Score " << getIrisScore(classifier, X, y) << std::endl;

  std::fstream out_stream("unittest.weightfile", std::ios_base::out | std::ios_base::trunc);
  out_stream << classifier << std::endl;
  out_stream.close();

  classifier.Print();

  std::fstream in_stream("unittest.weightfile", std::ios_base::in);
  FastBDT::Classifier classifier2(in_stream);

  std::cout << "Score " << getIrisScore(classifier2, X, y) << std::endl;

  classifier2.Print();

  return 0;
}
