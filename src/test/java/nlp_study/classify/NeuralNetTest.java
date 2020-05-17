package nlp_study.classify;

import org.junit.Test;
import static org.junit.Assert.*;
import static org.hamcrest.CoreMatchers.*;

import java.util.Collections;
import java.util.ArrayList;
import nlp_study.data.LabeledVector;



import nlp_study.data.LearningDataSet;

public class NeuralNetTest {
  @Test 
  public void testNaiveBayes() throws Exception{
    var dataSet = LearningDataSet.readFromFile("resources/person_data.txt");
    var lvList = dataSet.lebeledVecs;
    var trainingDataList = lvList.subList(0, lvList.size() - 1);
    var testData = lvList.get(lvList.size() - 1);
    var classifier = new NeuralClassifier();
    classifier.train(trainingDataList, dataSet.maxLabel, dataSet.maxFeature);

    assertThat(classifier.classify(testData.featureVector), is(testData.label));
  }

  @Test
  public void testBayesTestICM() throws Exception {
    int k = 10;
    var dataSet = LearningDataSet.readFromFile("resources/person_data.txt");
    var lvList = dataSet.lebeledVecs;
    Collections.shuffle(lvList);

    var confusionMatrix = new int[dataSet.maxLabel + 1][dataSet.maxFeature];
    int numCorrect = 0;

    for (int i = 0; i < k; ++i) {
      var trainingDataset = new ArrayList<LabeledVector>();
      var testDataset = new ArrayList<LabeledVector>();
      
      for(int j=0; j<lvList.size(); ++j) {
        if(j%k == i) {
          testDataset.add(lvList.get(j));
        }
        else {
          trainingDataset.add(lvList.get(j));
        }
      }

      var classifier = new NeuralClassifier();
      classifier.train(trainingDataset, dataSet.maxLabel, dataSet.maxFeature);

      for(var lv : testDataset){
        int c = classifier.classify(lv.featureVector);
        confusionMatrix[c][lv.label]++;
        if(c == lv.label) {
          numCorrect++;
        }
      }
    }

    System.out.println("*********************************** confusion matrix ***********************************");
    for(int i=1; i<dataSet.maxLabel; ++i) {
      for(int j=0; j<dataSet.maxLabel; ++j){
        System.out.println(confusionMatrix[i][j] + "/t");
      }
      System.out.println();
    }
    double correctProb = ((double) numCorrect) / lvList.size();
    assertTrue("correctProb is " + correctProb, correctProb > 0.8);
  }
}