package nlp_study.data;

import java.util.List;
import java.util.ArrayList;
import java.util.HashMap;
import java.nio.file.Paths;
import java.nio.file.Files;

public class LearningDataSet {
  public List< LabeledVector >  lebeledVecs;
  public int maxLabel;
  public int maxFeature;

  public static LearningDataSet readFromFile(String file) throws Exception{
    var dataSet = new LearningDataSet();
    dataSet.lebeledVecs = new ArrayList< LabeledVector >();
    dataSet.maxFeature = 0;
    dataSet.maxLabel = 0;

    Files.readAllLines(Paths.get(file)).stream()
      .map((str) -> {
        var lv = new LabeledVector();
        String[] split1 = str.split("[ \t]+");
        lv.label = Integer.parseInt(split1[0]);
        dataSet.maxLabel = dataSet.maxLabel > lv.label ? dataSet.maxLabel : lv.label;
        lv.name = split1[1];
        lv.featureVector = new HashMap<Integer, Double>();
        for(int i=2; i<split1.length; ++i){
          String[] split2 = split1[i].split(":");
          int feature = Integer.parseInt(split2[0]);
          double val = Double.parseDouble(split2[1]);
          lv.featureVector.put(feature, val);
          dataSet.maxFeature = feature > dataSet.maxFeature ? feature : dataSet.maxFeature;
        }
        return lv;
      })
      .forEach(dataSet.lebeledVecs::add);
    return dataSet;
  }
  
}