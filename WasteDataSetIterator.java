package ai.certifai.solution.classification;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class WasteDataSetIterator {
    private static int seed = 123;
    private static Random rng = new Random(seed);
    private static int width = 80;
    private static int height = 80;
    private static int nChannel= 3;
    private static double trainPerc;
    private static String [] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
    private static int batchSizeA;
    private static int numClass;
    private static ParentPathLabelGenerator myLabels = new ParentPathLabelGenerator();
    private static BalancedPathFilter balancedPathFilter = new BalancedPathFilter(rng, allowedExt, myLabels);
    private static ImageTransform imgTransform;
    static InputSplit trainData, testData;


    public static void setup (File file, int channel, int nClass, ImageTransform imageTransform, int batchSize, double trainTestRatio){
        numClass= nClass;
        batchSizeA = batchSize;
        nChannel = channel;
        imgTransform = imageTransform;
        trainPerc = trainTestRatio;

        FileSplit fileSplit = new FileSplit(file);

        if (trainPerc>1){
            throw new IllegalArgumentException("Train Percentage must be lower than 1");
        }

        InputSplit[] allData = fileSplit.sample(balancedPathFilter, trainPerc, 1-trainPerc);
        trainData = allData[0];
        testData = allData[1];
    }

    private static DataSetIterator makeIterator (InputSplit split, boolean training) throws IOException {
        ImageRecordReader imRR = new ImageRecordReader(height, width, nChannel, myLabels);
        if (training && imgTransform != null){
            imRR.initialize(trainData, imgTransform);
        }else {
            imRR.initialize(testData);
        }

        DataSetIterator iter = new RecordReaderDataSetIterator(imRR, batchSizeA, 1, numClass);

        DataNormalization scaler = new ImagePreProcessingScaler();
        iter.setPreProcessor(scaler);

        return iter;
    }

    public DataSetIterator trainIterator() throws IOException {
        DataSetIterator dataSetIterator = makeIterator(trainData, true);
        return dataSetIterator;
    }

    public DataSetIterator testIterator () throws IOException {
        return makeIterator(testData, false);
    }


}

