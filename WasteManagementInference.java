package ai.certifai.solution.classification;

import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import javax.swing.*;
import javax.swing.filechooser.FileSystemView;
import java.io.File;

public class WasteManagementInference {
    private static Logger log = LoggerFactory.getLogger(WasteManagementInference.class);
    private static final int width = 80;
    private static final int height = 80;
    public static String testImagePATH = "";
    private static MultiLayerNetwork model;
    private static File modelFilename = new File(System.getProperty("user.dir"), "generated-models/WasteClassifier.zip");

    public static void main(String[] args) throws Exception {

        JFrame frame = new JFrame("Waste Detector");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400, 400);

        JPanel panel = new JPanel();
        JLabel label = new JLabel("Upload waste image");
        JFileChooser jfc = new JFileChooser(FileSystemView.getFileSystemView().getHomeDirectory());

        int returnValue = jfc.showOpenDialog(null);
        // int returnValue = jfc.showSaveDialog(null);

        if (returnValue == JFileChooser.APPROVE_OPTION) {
            File selectedFile = jfc.getSelectedFile();
            System.out.println(selectedFile.getAbsolutePath());
            testImagePATH = selectedFile.getAbsolutePath();

            File imageToTest = new File(testImagePATH);
            model = ModelSerializer.restoreMultiLayerNetwork(modelFilename);

            NativeImageLoader loader = new NativeImageLoader(height, width, 3);
            INDArray image = loader.asMatrix(imageToTest);

            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
            scaler.transform(image);

            INDArray outputs = model.output(image);
            Nd4j.argMax(outputs, 1);
            log.info("Label:         " + Nd4j.argMax(outputs, 1));
            log.info("Probabilities: " + outputs.toString());

        }


    }
}
