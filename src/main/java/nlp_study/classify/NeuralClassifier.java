package nlp_study.classify;

import java.util.List;
import java.util.Map;
import java.util.Random;


import nlp_study.data.LabeledVector;


public class NeuralClassifier implements Classifier{
  public int maxLabel;
  public int maxFeature;
  public int numHiddenUnit;
  public double[][] w1;
  public double[][] w2;
  public double eta = 0.1;
  public int maxEpoch = 300;
  public double threashold = 0.01;

  public void train(List<LabeledVector> trainingDataset, int maxLabel, int maxFeature) {
    this.maxLabel = maxLabel;
    this.maxFeature = maxFeature;
    Random random = new Random();
    
    w1 = new double[numHiddenUnit+1][maxFeature+1];
    for(int i =1; i<numHiddenUnit; ++i) {
      for(int j=0; j<maxFeature; ++j) {
        w1[i][j] = random.nextDouble() - 0.5;
      } 
    }
    w2 = new double[maxLabel+1][numHiddenUnit+1];
    for(int i =0; i<maxLabel; ++i) {
      for(int j=1; j<numHiddenUnit; ++j) {
        w1[i][j] = random.nextDouble() - 0.5;
      } 
    }

    double[] inputUnits = new double[maxFeature+1];
    double[] hiddenUnits = new double[numHiddenUnit+1];
    double[] outUnits = new double[maxLabel+1];

    double[] answer = new double[maxLabel+1]; // 各出力層ユニットが出力するべき値
    double[] delta1 = new double[numHiddenUnit+1]; // 中間層の誤差
    double[] delta2 = new double[maxLabel+1]; // 出力層の誤差

    for(int epoch=1; epoch<=maxEpoch; ++epoch) {
      double err = 0.0; //平均二条誤差
      for(var lv: trainingDataset) {
        for(var entry: lv.featureVector.entrySet()) {
          inputUnits[entry.getKey()] = entry.getValue();
        }
        inputUnits[0] = 1.0; // バイアス項
        // 入力層から中間層へ
        for(int i=1; i<=numHiddenUnit; ++i) {
          double u = 0.0;
          for(int j=0; j<=maxFeature; ++j) {
            u += w1[i][j] * inputUnits[j];
          }
          hiddenUnits[i] = sigmoid(u);
        }
        hiddenUnits[0] = 1.0; // バイアス項
        // 中間層から出力層へ
        for(int i=1; i<=maxLabel; ++i) {
          double u = 0.0;
          for(int j=0; j<=numHiddenUnit; ++j) {
            u += w2[i][j] * hiddenUnits[j];
          }
          outUnits[i] = u;
        }
        // 誤差逆伝播学習 
        for(int i=1; i<=maxLabel; ++i) {
          answer[i] = i == lv.label ? 1.0 : 0.0;
        }
        for(int i=1; i<=maxLabel; ++i) {
          delta2[i] = (outUnits[i] - answer[i]) * (1.0 - outUnits[i])*outUnits[i];
          err = (outUnits[i]-answer[i])*(outUnits[i]-answer[i]);
        }
        for(int i=1; i<numHiddenUnit; ++i) {
          delta1[i] = 0.0;
          for(int j=1; j<maxLabel; ++j) {
            delta1[i] += w2[j][i] * delta2[j];
          }
          delta1[i] *= (1-hiddenUnits[i])*hiddenUnits[i];
        }
        for(int i=1; i<=maxLabel; ++i) {
          for(int j=0; j<=numHiddenUnit; ++j) {
            w2[i][j] -= eta*delta2[i] * hiddenUnits[j];
          }
        }
        for(int i=1; i<=numHiddenUnit; ++i) {
          for(int j=0; j<=maxFeature; ++j) {
            w1[i][j] -= eta*delta1[i] * inputUnits[j];
          }
        }
      }
      err /= 2;
      err /= trainingDataset.size();
      if(err < threashold) {
        break;
      }
    }
  }
  public int classify(Map<Integer, Double> featureVector){
    double[] inputUnits = new double[maxFeature+1];
    double[] hiddenUnits = new double[numHiddenUnit+1];
    double[] outputUnits = new double[maxLabel+1];

    for(var entry : featureVector.entrySet()) {
      inputUnits[entry.getKey()] = entry.getValue();
    }
    inputUnits[0] = 1.0;
    // 入力層から中間層へ
    for(int i=0; i<=numHiddenUnit; ++i) {
      double u = 0.0;
      for(int j=0; j<=maxFeature; ++j) {
        u += w1[i][j] * inputUnits[j];
      }
      hiddenUnits[i] = sigmoid(u);
    }
    hiddenUnits[0] = 1.0;
    // 中間層から出力層へ
    for(int i=1; i<=maxLabel; ++i) {
      double u = 0.0;
      for(int j=0; j<=numHiddenUnit; ++j) {
        u += w2[i][j] * hiddenUnits[j];
      }
      outputUnits[i] = sigmoid(u);
    }
    
    // 結果発表
    double maxProb = 0.0;
    int maxProbLabel = 0;
    for(int i=1; i<=maxLabel; ++i) {
      if(outputUnits[i] > maxProb) {
        maxProb = outputUnits[i];
        maxProbLabel = i;
      }
    }
    return maxProbLabel;

  }


 private double sigmoid(double x) {
   return 1.0 / (1.0 + Math.exp(-x));
 }
}