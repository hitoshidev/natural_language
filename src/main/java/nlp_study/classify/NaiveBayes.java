package nlp_study.classify;

import nlp_study.data.LabeledVector;
import java.util.List;
import java.util.Map;

public class NaiveBayes implements Classifier{
  public double alpha = 1.0;
  public int maxLabel;
  public int maxFeature;
  public int[] labelCount;
  public int[][] featureCount;
  public double[] labelProb;
  public double[][] featureProb;

  @Override
  public void train(final List<LabeledVector> trainingDataset, final int maxLabel, final int maxFeature){
    this.maxLabel = maxLabel;
    this.maxFeature = maxFeature;

    labelCount = new int[maxLabel+1];
    featureCount = new int[maxLabel+1][maxFeature+1];
    labelProb = new double[maxLabel+1];
    featureProb = new double[maxLabel+1][maxFeature+1];

    for(var lv : trainingDataset){
      labelCount[lv.label]++;
      for(var entry: lv.featureVector.entrySet()){
        if(entry.getValue() != 0) {
          featureCount[lv.label][entry.getKey()]++;
        }
      }
    }

    for(int i = 1; i<=maxLabel; ++i){
      labelProb[i] = (double) (labelCount[i] + alpha) / (trainingDataset.size()+alpha*maxLabel);
      for(int j=i; j<=maxFeature; ++j){
        featureProb[i][j] = (double) (featureCount[i][j]+alpha) / (labelCount[i] + alpha*2.0);
      }
    }
  }

  @Override
  public int classify(Map<Integer, Double> featureVector){
    double maxProb = 0.0;
    int maxProbLabel = 0;

    for(int i=1; i<=maxLabel; ++i) {
      double prob = labelProb[i];
      for(int j=1; j<=maxFeature; ++j) {
        var d = featureVector.get(j);
        if(d != null && d != 0.0) {
          prob *= featureProb[i][j];
        }
        else  {
          prob *= 1 - featureProb[i][j];
        }
      }
      if(prob > maxProb) {
        maxProb = prob;
        maxProbLabel = i;
      }
    }
    return maxProbLabel;
  }
}