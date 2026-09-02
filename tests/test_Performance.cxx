/**
 * Thomas Keck 2017
 */

#include "FastBDT.h"

#include <gtest/gtest.h>

#include <limits>
#include <chrono>
#include <random>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace FastBDT;

// Number of iterations of the performance tests
constexpr unsigned short TEST_ITERATIONS = 5;
// Number of times a failing measurement is repeated before it is reported
constexpr unsigned short TEST_ATTEMPTS = 3;
// Allow 2× for O(N log N) sort vs O(N) assumption
constexpr double ERROR_MARGIN = 2.0;

/**
 * These tests measure wall clock time, so they compete with whatever else runs
 * on the machine, which used to make them fail at random on CI. Interference is
 * one sided (it can only ever make a measurement slower) so we keep the
 * fastest of TEST_ITERATIONS interleaved timings and retry a failure up to
 * TEST_ATTEMPTS times.
 *
 * We compare marginal costs, the time for one more unit of work, rather than
 * averages, since every measurement carries a constant offset unrelated to the
 * quantity under test.
 */

// Checks that the marginal cost of one more unit of work does not grow with the
// size. If the runtime is linear every step costs the same, so each is compared
// against the median step; a fixed reference step would rescale every ratio
// whenever that one measurement happens to be disturbed. Returns an empty
// string on success, a description of the problem otherwise.
static std::string CheckLinearScaling(const std::vector<unsigned int>& sizes, const std::vector<double>& times)
{
  // marginal[i] is the cost of one unit of work over the step sizes[i] -> sizes[i + 1]
  std::vector<double> marginal;
  for (unsigned int i = 1; i < sizes.size(); ++i)
    marginal.push_back((times[i] - times[i - 1]) / static_cast<double>(sizes[i] - sizes[i - 1]));

  std::ostringstream failure;

  // A step that came out free, or negative, measured interference and not work
  for (unsigned int i = 0; i < marginal.size(); ++i)
    if (marginal[i] <= 0.0)
      failure << "the step up to size " << sizes[i + 1] << " came out at " << marginal[i]
              << "us per unit of work, so the measurement is unusable; ";
  if (!failure.str().empty())
    return failure.str();

  std::vector<double> sorted = marginal;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t middle = sorted.size() / 2;
  const double reference = sorted.size() % 2 == 0 ? 0.5 * (sorted[middle - 1] + sorted[middle]) : sorted[middle];

  for (unsigned int i = 0; i < marginal.size(); ++i) {
    double ratio = marginal[i] / reference;
    std::cout << "Marginal cost up to size " << sizes[i + 1] << ": " << marginal[i]
              << "us per unit of work, ratio to the typical step: " << ratio << std::endl;
    if (ratio >= ERROR_MARGIN)
      failure << "one more unit of work costs " << ratio << " times as much at size " << sizes[i + 1]
              << " as it does in a typical step (allowed " << ERROR_MARGIN << "); ";
  }
  return failure.str();
}

// Times every entry of sizes and asserts linear scaling, retrying the complete
// measurement while it fails.
template<class Measure>
static void ExpectLinearScaling(const std::vector<unsigned int>& sizes, Measure&& measure)
{
  std::string failure;
  for (unsigned short attempt = 1; attempt <= TEST_ATTEMPTS; ++attempt) {
    // Sweep all sizes once per iteration instead of finishing one size before
    // starting the next, so that each size gets its samples spread over the whole
    // run. Otherwise a burst of interference covers every sample of whichever
    // size is being measured at the time, and keeping the fastest one protects
    // nothing. The sweep goes small to large, so those bursts would land on the
    // large sizes and look exactly like superlinear scaling.
    std::vector<double> times(sizes.size(), std::numeric_limits<double>::max());
    for (unsigned short i = 0; i < TEST_ITERATIONS; ++i)
      for (unsigned int j = 0; j < sizes.size(); ++j)
        times[j] = std::min(times[j], measure(sizes[j]));

    failure = CheckLinearScaling(sizes, times);
    if (failure.empty())
      return;

    std::cout << "Attempt " << attempt << " of " << TEST_ATTEMPTS << " failed: " << failure
              << (attempt < TEST_ATTEMPTS ? "retrying" : "giving up") << std::endl;
  }
  ADD_FAILURE() << "Failed in all " << TEST_ATTEMPTS << " attempts: " << failure;
}

class PerformanceFeatureBinningTest : public ::testing::Test {
protected:
  virtual void SetUp()
  {
    std::default_random_engine generator(42);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    unsigned int N = 10000000;
    data.resize(N);
    for (unsigned int i = 0; i < N; ++i) {
      data[i] = distribution(generator);
    }
  }
  std::vector<float> data;
};

TEST_F(PerformanceFeatureBinningTest, FeatureBinningScalesLinearInNumberOfDataPoints)
{
  // This is dominated by the sorting of the numbers -> N log (N),
  // for our purposes we assume just N, which seems to be fine
  // if this unittest starts failing I have to revise this and add the factor of log(N)
  std::vector<unsigned int> sizes = {1000, 10000, 100000, 1000000};

  auto measure = [this](unsigned int size) {
    std::vector<float> temp_data(data.begin(), data.begin() + size);

    auto start = std::chrono::high_resolution_clock::now();
    FeatureBinning<float> binning(4, temp_data);
    auto stop = std::chrono::high_resolution_clock::now();

    // We check something simple, so that we are sure that the compiler cannot optimize out the binning itself
    EXPECT_EQ(binning.GetNLevels(), 4u);

    return std::chrono::duration<double, std::micro>(stop - start).count();
  };

  ExpectLinearScaling(sizes, measure);
}

TEST_F(PerformanceFeatureBinningTest, FeatureBinningScalesConstantInSmallNumberOfLayers)
{
  // The feature binning should be dominated by the sorting of the numbers
  // hence it does not scale with the number of layers to first order
  // for large layers this will be wrong ~ #Layer > 17
  std::vector<unsigned int> sizes = {2, 3, 5, 7, 11, 13, 17};
  std::vector<double> times;

  for (auto& size : sizes) {

    // Repeat the test few times and calculate the average time
    double temp_time = 0.0;
    for (unsigned short i = 0; i < TEST_ITERATIONS; ++i) {

      std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
      FeatureBinning<float> binning(size, data);
      std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();

      // We check something simple, so that we are sure that the compiler cannot optimize out the binning itself
      EXPECT_EQ(binning.GetNLevels(), size);

      std::chrono::duration<double, std::micro> time = stop - start;
      temp_time += time.count();
    }

    times.push_back(temp_time / TEST_ITERATIONS);
  }

  // Check linear behaviour
  // We ignore the first measurement, to avoid effects of caching
  for (unsigned int i = 1; i < sizes.size(); ++i) {
    double time_ratio = times[i] / static_cast<double>(times[1]);
    EXPECT_GT(time_ratio,  0.8);
    EXPECT_LT(time_ratio,  1.2);
  }
}

class PerformanceTreeBuilderTest : public ::testing::Test {
protected:
  std::default_random_engine generator;
  std::uniform_int_distribution<unsigned int> distribution{0, 16};
};

TEST_F(PerformanceTreeBuilderTest, TreeBuilderScalesLinearInNumberOfDataPoints)
{
  auto random_source = std::bind(distribution, generator);

  unsigned int nFeatures = 10;
  unsigned int nLayers = 4;

  std::vector<unsigned int> sizes = {1000, 10000, 100000, 1000000, 10000000};

  auto measure = [&](unsigned int nDataPoints) {
    std::vector<unsigned int> row(nFeatures);
    std::vector<unsigned int> binning_levels(nFeatures, 4);

    EventSample sample(nDataPoints, nFeatures, 0, binning_levels);
    for (unsigned int j = 0; j < nDataPoints; ++j) {
      std::generate_n(row.begin(), nFeatures, random_source);
      sample.AddEvent(row, 1.0, j % 2 == 0);
    }

    auto start = std::chrono::high_resolution_clock::now();
    TreeBuilder dt(nLayers, sample);
    auto stop = std::chrono::high_resolution_clock::now();

    // We check something simple, so that we are sure that the compiler cannot optimize out the binning itself
    const auto& purities = dt.GetPurities();
    EXPECT_EQ(purities.size(), static_cast<unsigned int>((1 << (nLayers + 1)) - 1));

    return std::chrono::duration<double, std::micro>(stop - start).count();
  };

  ExpectLinearScaling(sizes, measure);
}

TEST_F(PerformanceTreeBuilderTest, TreeBuilderScalesLinearInNumberOfFeatures)
{
  auto random_source = std::bind(distribution, generator);

  unsigned int nLayers = 4;
  unsigned int nDataPoints = 100000;

  // Below ~8 features the per-event work dominates and the steps become small
  // differences of large numbers; at 512 features the data no longer fits near
  // the cache. Both cost a constant factor that says nothing about the scaling.
  std::vector<unsigned int> sizes = {8, 16, 32, 64, 128, 256};

  auto measure = [&](unsigned int nFeatures) {
    std::vector<unsigned int> row(nFeatures);
    std::vector<unsigned int> binning_levels(nFeatures, 4);

    EventSample sample(nDataPoints, nFeatures, 0, binning_levels);
    for (unsigned int j = 0; j < nDataPoints; ++j) {
      std::generate_n(row.begin(), nFeatures, random_source);
      sample.AddEvent(row, 1.0, j % 2 == 0);
    }

    auto start = std::chrono::high_resolution_clock::now();
    TreeBuilder dt(nLayers, sample);
    auto stop = std::chrono::high_resolution_clock::now();

    // We check something simple, so that we are sure that the compiler cannot optimize out the binning itself
    const auto& purities = dt.GetPurities();
    EXPECT_EQ(purities.size(), static_cast<unsigned int>((1 << (nLayers + 1)) - 1));

    return std::chrono::duration<double, std::micro>(stop - start).count();
  };

  ExpectLinearScaling(sizes, measure);
}

TEST_F(PerformanceTreeBuilderTest, TreeBuilderScalesLinearForSmallNumberOfLayers)
{
  // For small numbers of layers (below 10) we should scale linear,
  // above the number of nodes in the deeper layers of the tree gets in the same order
  // of magnitude as the number of data_points and the summing of the histograms
  // becomes important
  auto random_source = std::bind(distribution, generator);

  unsigned int nFeatures = 10;
  unsigned int nDataPoints = 100000;

  // Past 7 layers the cost per layer climbs, the number of nodes grows like 2^nLayers
  std::vector<unsigned int> sizes = {1, 2, 3, 5, 7};

  std::vector<unsigned int> row(nFeatures);
  std::vector<unsigned int> binning_levels(nFeatures, 4);
  EventSample sample(nDataPoints, nFeatures, 0, binning_levels);
  for (unsigned int i = 0; i < nDataPoints; ++i) {
    std::generate_n(row.begin(), nFeatures, random_source);
    sample.AddEvent(row, 1.0, i % 2 == 0);
  }

  auto measure = [&](unsigned int nLayers) {
    // Copy the sample for this iteration so TreeBuilder can mutate it safely
    EventSample sample_copy = sample;

    // Reset flags, so we can use the sample multiple times
    auto& flags = sample_copy.GetFlags();
    for (unsigned int iEvent = 0; iEvent < nDataPoints; ++iEvent)
      flags.Set(iEvent, 1);

    auto start = std::chrono::high_resolution_clock::now();
    TreeBuilder dt(nLayers, sample_copy);
    auto stop = std::chrono::high_resolution_clock::now();

    // We check something simple, so that we are sure that the compiler cannot optimize out the binning itself
    const auto& purities = dt.GetPurities();
    EXPECT_EQ(purities.size(), static_cast<unsigned int>((1 << (nLayers + 1)) - 1));

    return std::chrono::duration<double, std::micro>(stop - start).count();
  };

  ExpectLinearScaling(sizes, measure);
}
