package nlp_study.classify;

import nlp_study.data.LabeledVector;
import java.util.List;
import java.util.Map;

public interface Classifier {
  public void train(List<LabeledVector> trainingDataset, int maxLabel, int maxFeature);

  public int classify(Map<Integer, Double> featureVec );
}