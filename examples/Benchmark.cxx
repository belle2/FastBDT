/**
 * FastBDT Performance Benchmark
 *
 * Measures training time, inference throughput, and serialisation speed
 * across different dataset sizes using the high-level Classifier interface.
 * Run before and after significant changes to catch performance regressions.
 */

#include <Classifier.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

struct Stats { double mean, stddev, min, max; };

static Stats computeStats(const std::vector<double>& v)
{
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / static_cast<double>(v.size());
  double sq = 0.0, mn = v[0], mx = v[0];
  for (double x : v) {
    sq += (x - mean) * (x - mean);
    if (x < mn) mn = x;
    if (x > mx) mx = x;
  }
  return {mean, std::sqrt(sq / static_cast<double>(v.size())), mn, mx};
}

static void printHeader(const std::string& title)
{
  const std::string bar(64, '=');
  std::cout << "\n" << bar << "\n  " << title << "\n" << std::string(64, '-') << "\n";
  std::cout << std::left  << std::setw(18) << "dataset size"
            << std::right << std::setw(10) << "mean"
            << std::setw(10) << "stddev"
            << std::setw(10) << "min"
            << std::setw(10) << "max"
            << "  unit\n"
            << std::string(64, '-') << "\n";
}

static void printRow(const std::string& label, const Stats& s, const std::string& unit)
{
  std::cout << std::left  << std::setw(18) << label
            << std::right << std::fixed << std::setprecision(2)
            << std::setw(10) << s.mean
            << std::setw(10) << s.stddev
            << std::setw(10) << s.min
            << std::setw(10) << s.max
            << "  " << unit << "\n";
}

static std::string humanSize(unsigned n)
{
  if (n >= 1000000) return std::to_string(n / 1000000) + "M";
  if (n >= 1000)    return std::to_string(n / 1000)    + "k";
  return std::to_string(n);
}

// ---------------------------------------------------------------------------
// Synthetic data generation
// ---------------------------------------------------------------------------
//
// Produces `nFeatures` Gaussian features. Signal events (even indices) are
// drawn from N(+0.5, 1), background from N(-0.5, 1), giving moderate but
// non-trivial separability. All weights are 1.
// X is column-major: X[iFeature][iEvent], matching Classifier::fit convention.

static void generateData(unsigned nEvents, unsigned nFeatures, unsigned seed,
                         std::vector<std::vector<float>>& X,
                         std::vector<bool>&               y,
                         std::vector<FastBDT::Weight>&    w)
{
  std::mt19937 rng(seed);
  std::normal_distribution<float> sig(+0.5f, 1.0f);
  std::normal_distribution<float> bkg(-0.5f, 1.0f);

  X.assign(nFeatures, std::vector<float>(nEvents));
  y.assign(nEvents, false);
  w.assign(nEvents, static_cast<FastBDT::Weight>(1));

  for (unsigned iEvent = 0; iEvent < nEvents; ++iEvent) {
    bool isSignal = (iEvent % 2 == 0);
    y[iEvent] = isSignal;
    auto& dist = isSignal ? sig : bkg;
    for (unsigned iFeat = 0; iFeat < nFeatures; ++iFeat)
      X[iFeat][iEvent] = dist(rng);
  }
}

// ---------------------------------------------------------------------------
// Classifier factory — fixed hyperparameters matching library defaults
// ---------------------------------------------------------------------------

static FastBDT::Classifier makeClassifier(unsigned nFeatures,
                                          unsigned nTrees  = 100,
                                          unsigned depth   = 3,
                                          unsigned nLevels = 8)
{
  FastBDT::Classifier clf;
  clf.SetNTrees(nTrees);
  clf.SetDepth(depth);
  clf.SetShrinkage(0.1);
  clf.SetSubsample(0.5);
  clf.SetSPlot(false);
  clf.SetTransform2Probability(true);
  clf.SetBinning(std::vector<unsigned int>(nFeatures, nLevels));
  clf.SetPurityTransformation(std::vector<bool>(nFeatures, false));
  clf.SetNumberOfFlatnessFeatures(0);
  clf.SetFlatnessLoss(-1.0);
  return clf;
}

// ---------------------------------------------------------------------------
// Benchmark 1: Training time vs dataset size
// ---------------------------------------------------------------------------

static void benchmarkTraining(unsigned nRepeat)
{
  const unsigned nFeatures = 10;
  printHeader("Training  (nFeatures=10, nTrees=100, depth=3, nLevels=8)");

  const std::vector<unsigned> sizes = {1000, 10000, 100000, 1000000};

  for (unsigned nEvents : sizes) {
    std::vector<std::vector<float>> X;
    std::vector<bool>               y;
    std::vector<FastBDT::Weight>    w;
    generateData(nEvents, nFeatures, 42, X, y, w);

    std::vector<double> times;
    times.reserve(nRepeat);
    for (unsigned rep = 0; rep < nRepeat; ++rep) {
      auto clf = makeClassifier(nFeatures);
      auto t0 = Clock::now();
      clf.fit(X, y, w);
      auto t1 = Clock::now();
      times.push_back(Ms(t1 - t0).count());
    }

    printRow(humanSize(nEvents) + " events", computeStats(times), "ms");
  }
}

// ---------------------------------------------------------------------------
// Benchmark 2: Inference throughput (events/second)
// ---------------------------------------------------------------------------

static void benchmarkInference(unsigned nRepeat)
{
  const unsigned nFeatures    = 10;
  const unsigned nTrainEvents = 100000;

  // Train once
  std::vector<std::vector<float>> Xtrain;
  std::vector<bool>               ytrain;
  std::vector<FastBDT::Weight>    wtrain;
  generateData(nTrainEvents, nFeatures, 42, Xtrain, ytrain, wtrain);
  auto clf = makeClassifier(nFeatures);
  clf.fit(Xtrain, ytrain, wtrain);

  printHeader("Inference (trained on 100k events, 10 features)");

  const std::vector<unsigned> sizes = {1000, 10000, 100000, 1000000};

  for (unsigned nEvents : sizes) {
    // Build row-major test data: rows[iEvent] = {feat0, feat1, ...}
    std::vector<std::vector<float>> Xtest;
    std::vector<bool>               ytest;
    std::vector<FastBDT::Weight>    wtest;
    generateData(nEvents, nFeatures, 99, Xtest, ytest, wtest);

    std::vector<std::vector<float>> rows(nEvents, std::vector<float>(nFeatures));
    for (unsigned iEvent = 0; iEvent < nEvents; ++iEvent)
      for (unsigned iFeat = 0; iFeat < nFeatures; ++iFeat)
        rows[iEvent][iFeat] = Xtest[iFeat][iEvent];

    std::vector<double> times;
    times.reserve(nRepeat);
    double checksum = 0.0; // prevents dead-code elimination of predict()
    for (unsigned rep = 0; rep < nRepeat; ++rep) {
      double sum = 0.0;
      auto t0 = Clock::now();
      for (unsigned iEvent = 0; iEvent < nEvents; ++iEvent)
        sum += clf.predict(rows[iEvent]);
      auto t1 = Clock::now();
      checksum += sum;
      times.push_back(Ms(t1 - t0).count());
    }

    printRow(humanSize(nEvents) + " events", computeStats(times), "ms");
    // Print checksum so the compiler cannot eliminate the loop above
    std::cout << "  (checksum=" << std::scientific << std::setprecision(4)
              << checksum << ")\n";
  }
}

// ---------------------------------------------------------------------------
// Benchmark 3: Serialisation (save/load via stream)
// ---------------------------------------------------------------------------

static void benchmarkSerialisation(unsigned nRepeat)
{
  const unsigned nFeatures    = 10;
  const unsigned nTrainEvents = 100000;

  std::vector<std::vector<float>> X;
  std::vector<bool>               y;
  std::vector<FastBDT::Weight>    w;
  generateData(nTrainEvents, nFeatures, 42, X, y, w);
  auto clf = makeClassifier(nFeatures);
  clf.fit(X, y, w);

  // Pre-serialise once to get the string for the load benchmark
  std::ostringstream oss;
  oss << clf;
  const std::string serialised = oss.str();

  printHeader("Serialisation  (100k training events, 10 features, 100 trees)");

  // Save
  {
    std::vector<double> times;
    times.reserve(nRepeat);
    for (unsigned rep = 0; rep < nRepeat; ++rep) {
      std::ostringstream buf;
      auto t0 = Clock::now();
      buf << clf;
      auto t1 = Clock::now();
      times.push_back(Ms(t1 - t0).count());
    }
    printRow("save (ostream)", computeStats(times), "ms");
  }

  // Load
  {
    std::vector<double> times;
    times.reserve(nRepeat);
    double checksum = 0.0;
    for (unsigned rep = 0; rep < nRepeat; ++rep) {
      std::istringstream buf(serialised);
      auto t0 = Clock::now();
      FastBDT::Classifier clf2(buf);
      auto t1 = Clock::now();
      checksum += clf2.GetNTrees();
      times.push_back(Ms(t1 - t0).count());
    }
    printRow("load (istream)", computeStats(times), "ms");
    std::cout << "  (checksum=" << checksum << ")\n";
  }

  std::cout << "  serialised size: " << serialised.size() << " bytes\n";
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
  unsigned nRepeat = 5;
  if (argc >= 2) {
    try {
      nRepeat = static_cast<unsigned>(std::stoul(argv[1]));
    } catch (...) {
      std::cerr << "Usage: " << argv[0] << " [nRepeat>=2]\n";
      return 1;
    }
  }
  if (nRepeat < 2) {
    std::cerr << "nRepeat must be >= 2\n";
    return 1;
  }

  std::cout << "FastBDT Benchmark  (nRepeat=" << nRepeat << ")\n";

  benchmarkTraining(nRepeat);
  benchmarkInference(nRepeat);
  benchmarkSerialisation(nRepeat);

  std::cout << "\n" << std::string(64, '=') << "\n";
  return 0;
}
