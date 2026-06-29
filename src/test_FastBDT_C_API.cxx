/**
 * Thomas Keck 2015
 */

#include "FastBDT_C_API.h"

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>

using namespace FastBDT;

class CInterfaceTest : public ::testing::Test {
protected:
  virtual void SetUp()
  {
    expertise = static_cast<Expertise*>(Create());
  }

  virtual void TearDown()
  {
    Delete(expertise);
  }

  Expertise* expertise;

};

TEST_F(CInterfaceTest, SetGetBinning)
{

  unsigned int binning[] = {10u, 20u};
  SetBinning(expertise, binning, 2);
  EXPECT_EQ(expertise->classifier.GetBinning().size(), 2u);
  EXPECT_EQ(expertise->classifier.GetBinning()[0], 10u);
  EXPECT_EQ(expertise->classifier.GetBinning()[1], 20u);

}

TEST_F(CInterfaceTest, SetGetPurityTransformation)
{

  bool purityTransformation[] = {true, false};
  SetPurityTransformation(expertise, purityTransformation, 2);
  EXPECT_EQ(expertise->classifier.GetPurityTransformation().size(), 2u);
  EXPECT_EQ(expertise->classifier.GetPurityTransformation()[0], true);
  EXPECT_EQ(expertise->classifier.GetPurityTransformation()[1], false);

}

TEST_F(CInterfaceTest, SetGetNTrees)
{

  SetNTrees(expertise, 200u);
  EXPECT_EQ(expertise->classifier.GetNTrees(), 200u);

}

TEST_F(CInterfaceTest, SetGetSPlot)
{

  SetSPlot(expertise, false);
  EXPECT_EQ(expertise->classifier.GetSPlot(), false);
  SetSPlot(expertise, true);
  EXPECT_EQ(expertise->classifier.GetSPlot(), true);

}

TEST_F(CInterfaceTest, SetGetTransform2Probability)
{

  SetTransform2Probability(expertise, false);
  EXPECT_EQ(expertise->classifier.GetTransform2Probability(), false);
  SetTransform2Probability(expertise, true);
  EXPECT_EQ(expertise->classifier.GetTransform2Probability(), true);

}

TEST_F(CInterfaceTest, SetGetDepth)
{

  SetDepth(expertise, 5u);
  EXPECT_EQ(expertise->classifier.GetDepth(), 5u);
  SetDepth(expertise, 2u);
  EXPECT_EQ(expertise->classifier.GetDepth(), 2u);

}

TEST_F(CInterfaceTest, SetGetFlatnessLossWorks)
{

  SetFlatnessLoss(expertise, 0.2);
  EXPECT_DOUBLE_EQ(expertise->classifier.GetFlatnessLoss(), 0.2);
  SetFlatnessLoss(expertise, 0.4);
  EXPECT_DOUBLE_EQ(expertise->classifier.GetFlatnessLoss(), 0.4);

}

TEST_F(CInterfaceTest, SetGetShrinkageWorks)
{

  SetShrinkage(expertise, 0.2);
  EXPECT_DOUBLE_EQ(expertise->classifier.GetShrinkage(), 0.2);
  SetShrinkage(expertise, 0.4);
  EXPECT_DOUBLE_EQ(expertise->classifier.GetShrinkage(), 0.4);

}


TEST_F(CInterfaceTest, SetSubsampleWorks)
{

  SetSubsample(expertise, 0.6);
  EXPECT_DOUBLE_EQ(expertise->classifier.GetSubsample(), 0.6);
  SetSubsample(expertise, 0.8);
  EXPECT_DOUBLE_EQ(expertise->classifier.GetSubsample(), 0.8);

}


TEST_F(CInterfaceTest, FitAndPredictWorksWithoutWeights)
{

  // Use just one branch instead of a whole forest for testing
  // We only test if the ForestBuilder is called correctly,
  // the builder itself is tested elsewhere.
  SetNTrees(expertise, 10u);
  SetDepth(expertise, 1u);
  SetSubsample(expertise, 1.0);
  SetShrinkage(expertise, 1.0);
  unsigned int binning[] = {2u, 2u};
  SetBinning(expertise, binning, 2);
  SetTransform2Probability(expertise, true);
  SetNumberOfFlatnessFeatures(expertise, 0);

  float data_ptr[] = {1.0, 2.6, 1.6, 2.5, 1.1, 2.0, 1.9, 2.1, 1.6, 2.9, 1.9, 2.9, 1.5, 2.0};
  bool target_ptr[] = {0, 1, 0, 1, 1, 1, 0};
  Fit(expertise, data_ptr, nullptr, target_ptr, 7, 2);

  float test_ptr[] = {1.0, 2.6};
  EXPECT_LE(Predict(expertise, test_ptr), 0.01);

  float test_ptr2[] = {1.6, 2.5};
  EXPECT_GE(Predict(expertise, test_ptr2), 0.99);
}


TEST_F(CInterfaceTest, TrainAndAnalyseForestWorksWithSpectators)
{

  // Use just one branch instead of a whole forest for testing
  // We only test if the ForestBuilder is called correctly,
  // the builder itself is tested elsewhere.
  SetNTrees(expertise, 10u);
  SetDepth(expertise, 1u);
  SetSubsample(expertise, 1.0);
  SetShrinkage(expertise, 1.0);
  unsigned int binning[] = {2u, 2u, 2u, 3u};
  SetBinning(expertise, binning, 4);
  SetTransform2Probability(expertise, true);
  SetNumberOfFlatnessFeatures(expertise, 2);

  float data_ptr[] = {1.0, 2.6, 0.0, -10.0,
                      1.6, 2.5, 99.0, 0.0,
                      1.1, 2.0, -500.0, 12.1,
                      1.9, 2.1, 0.0, 0.0,
                      1.6, 2.9, 23.0, 42.0,
                      1.9, 2.9, 0.0, 1.0,
                      1.5, 2.0, 1.0, -1.0
                     };
  bool target_ptr[] = {0, 1, 0, 1, 1, 1, 0};
  Fit(expertise, data_ptr, nullptr, target_ptr, 7, 4);

  float test_ptr[] = {1.0, 2.6};
  EXPECT_LE(Predict(expertise, test_ptr), 0.03);
}

TEST_F(CInterfaceTest, TrainAndAnalyseForestWorksWithWeights)
{

  // Use just one branch instead of a whole forest for testing
  // We only test if the ForestBuilder is called correctly,
  // the builder itself is tested elsewhere.
  SetNTrees(expertise, 10u);
  SetDepth(expertise, 1u);
  SetSubsample(expertise, 1.0);
  SetShrinkage(expertise, 1.0);
  unsigned int binning[] = {2u, 2u};
  SetBinning(expertise, binning, 2);
  SetTransform2Probability(expertise, true);
  SetNumberOfFlatnessFeatures(expertise, 0);

  float data_ptr[] = {1.0, 2.6, 1.6, 2.5, 1.1, 2.0, 1.9, 2.1, 1.6, 2.9, 1.9, 2.9, 1.5, 2.0};
  Weight weight_ptr[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  bool target_ptr[] = {0, 1, 0, 1, 1, 1, 0};
  Fit(expertise, data_ptr, weight_ptr, target_ptr, 7, 2);

  float test_ptr[] = {1.0, 2.6};
  EXPECT_LE(Predict(expertise, test_ptr), 0.01);

  Weight weight_ptr2[] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  Fit(expertise, data_ptr, weight_ptr2, target_ptr, 7, 2);
  EXPECT_LE(Predict(expertise, test_ptr), 0.01);

  Weight weight_ptr3[] = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0};
  Fit(expertise, data_ptr, weight_ptr3, target_ptr, 7, 2);
  EXPECT_LE(Predict(expertise, test_ptr), 0.03);
}

// ---------------------------------------------------------------------------
// Fixture with a pre-trained classifier to share setup across several tests
// ---------------------------------------------------------------------------

class TrainedCInterfaceTest : public CInterfaceTest {
protected:
  virtual void SetUp() override
  {
    CInterfaceTest::SetUp();
    SetNTrees(expertise, 10u);
    SetDepth(expertise, 1u);
    SetSubsample(expertise, 1.0);   // deterministic: no random subsampling
    SetShrinkage(expertise, 1.0);
    unsigned int binning[] = {2u, 2u};
    SetBinning(expertise, binning, 2);
    SetTransform2Probability(expertise, true);
    SetNumberOfFlatnessFeatures(expertise, 0);
    Fit(expertise, data_ptr, nullptr, target_ptr, nEvents, nFeatures);
  }

  // 7 events x 2 features, row-major
  float data_ptr[14] = {1.0, 2.6, 1.6, 2.5, 1.1, 2.0, 1.9, 2.1, 1.6, 2.9, 1.9, 2.9, 1.5, 2.0};
  bool  target_ptr[7] = {0, 1, 0, 1, 1, 1, 0};
  const unsigned int nEvents   = 7;
  const unsigned int nFeatures = 2;
};

TEST_F(TrainedCInterfaceTest, PredictArrayMatchesSinglePredict)
{
  float results[7];
  PredictArray(expertise, data_ptr, results, nEvents);

  for (unsigned int i = 0; i < nEvents; ++i) {
    float single = Predict(expertise, &data_ptr[i * nFeatures]);
    EXPECT_FLOAT_EQ(results[i], single);
  }
}

TEST_F(TrainedCInterfaceTest, GetVariableRankingWorks)
{
  void* ranking = GetVariableRanking(expertise);
  ASSERT_NE(ranking, nullptr);

  unsigned int nVars = ExtractNumberOfVariablesFromVariableRanking(ranking);
  EXPECT_GE(nVars, 1u);

  for (unsigned int i = 0; i < nVars; ++i) {
    double importance = ExtractImportanceOfVariableFromVariableRanking(ranking, i);
    EXPECT_GE(importance, 0.0);
    EXPECT_LE(importance, 1.0);
  }

  DeleteVariableRanking(ranking);
}

TEST_F(TrainedCInterfaceTest, GetIndividualVariableRankingWorks)
{
  float test_ptr[] = {1.0f, 2.6f};
  void* ranking = GetIndividualVariableRanking(expertise, test_ptr);
  ASSERT_NE(ranking, nullptr);

  unsigned int nVars = ExtractNumberOfVariablesFromVariableRanking(ranking);
  EXPECT_GE(nVars, 1u);

  for (unsigned int i = 0; i < nVars; ++i) {
    double importance = ExtractImportanceOfVariableFromVariableRanking(ranking, i);
    EXPECT_GE(importance, 0.0);
    EXPECT_LE(importance, 1.0);
  }

  DeleteVariableRanking(ranking);
}

TEST_F(TrainedCInterfaceTest, SaveAndLoadWorks)
{
  float test_ptr[] = {1.0f, 2.6f};
  float score_before = Predict(expertise, test_ptr);

  const char* tmpfile = "/tmp/fastbdt_c_api_test.weightfile";
  Save(expertise, const_cast<char*>(tmpfile));

  void* expertise2 = Create();
  Load(expertise2, const_cast<char*>(tmpfile));
  float score_after = Predict(expertise2, test_ptr);
  Delete(expertise2);

  std::remove(tmpfile);

  EXPECT_FLOAT_EQ(score_before, score_after);
}

TEST_F(CInterfaceTest, PredictBeforeFitDoesNotCrash)
{
  // Before Fit, GetNFeatures() == 0, so Predict reads zero features and calls
  // Analyse on the default-constructed forest. Should return a finite value.
  float dummy[] = {1.0f, 2.0f};
  float result = Predict(expertise, dummy);
  EXPECT_FALSE(std::isnan(result));
  EXPECT_TRUE(std::isfinite(result));
}
