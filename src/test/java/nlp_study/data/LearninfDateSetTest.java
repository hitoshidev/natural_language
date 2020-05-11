package nlp_study.data;

import static org.junit.Assert.*;
import org.junit.Test;
import static org.hamcrest.CoreMatchers.*;

public class LearninfDateSetTest{
  @Test
  public void testReadFromFile()  throws Exception{
    String file = "resources/person_data.txt";
    var dataSet = LearningDataSet.readFromFile(file);
    assertThat(dataSet.maxLabel , is(2));
    assertThat(dataSet.maxFeature , is(7));
  }
}
