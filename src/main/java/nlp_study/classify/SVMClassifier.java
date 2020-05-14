package nlp_study.classify;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.Collections;

import libsvm.svm;
import libsvm.svm_parameter;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_problem;

import nlp_study.data.LabeledVector;

public class SVMClassifier implements Classifier{
  public svm_parameter parameter;
  public svm_model model;

  public SVMClassifier() {
    parameter = new svm_parameter();

    parameter.svm_type = svm_parameter.C_SVC;
    parameter.C = 1.0;
    parameter.kernel_type = svm_parameter.LINEAR;
    parameter.gamma = 0;
    parameter.degree = 2;
    parameter.coef0 = 0;
    parameter.cache_size = 100;
    parameter.eps = 1e-3;
  }

  @Override
  public void train(List<LabeledVector> trainingDataset, int maxLabel, int maxFeature) {
    svm_problem prob = new svm_problem();
    prob.l = trainingDataset.size();
    prob.x = new svm_node[prob.l][];
    prob.y = new double[prob.l];

    for(int i=0; i < trainingDataset.size(); ++i) {
      prob.x[i] = toSVMNode(trainingDataset.get(i).featureVector);
      prob.y[i] = (double) trainingDataset.get(i).label;
    }

    model = svm.svm_train(prob, parameter);
  }

  @Override
  public int classify(Map<Integer, Double> featureVec ) {
    svm_node[] nodes = toSVMNode(featureVec);
    double x = svm.svm_predict(model, nodes);
    return (int) x;
  }

  private svm_node[] toSVMNode(Map<Integer, Double> featureVector) {
    var indexList = new ArrayList< Integer >(featureVector.keySet());
    Collections.sort(indexList);
    svm_node[] nodes = new svm_node[indexList.size()];
    for(int i=0; i<indexList.size(); ++i) {
      nodes[i] = new svm_node();
      nodes[i].index = indexList.get(i);
      nodes[i].value = featureVector.get(indexList.get(i));
    }
    return nodes;
  }
}