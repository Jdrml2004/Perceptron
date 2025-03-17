import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

/**
 * The Main class serves as the entry point for the neural network application.
 * It provides options to train the neural network, test it with saved weights, or perform a specific operation called "mooshake."
 */
public class Main {

    /**
     * The main method initializes the Main instance and invokes the mooshake method.
     *
     * @param args Command-line arguments (not used).
     */
    public static void main(String[] args) {
        Main mainInstance = new Main();
        mainInstance.client();
    }

    /**
     * Presents a menu to the user to either train the neural network or test it using saved weights.
     */
    public void client() {
        Scanner scanner = new Scanner(System.in);

        System.out.println("Choose an option:");
        System.out.println("1 - Train the neural network");
        System.out.println("2 - Test the neural network with saved weights");
        int choice = scanner.nextInt();

        if (choice == 1) {
            exercise5(true);
        } else if (choice == 2) {
            exercise5(false);
        } else {
            System.out.println("Invalid option.");
        }

        scanner.close();
    }

    /**
     * Handles the training and testing of the neural network based on the provided flag.
     *
     * @param shouldTrain If true, the network will be trained; otherwise, it will load existing weights.
     */
    public void exercise5(boolean shouldTrain) {
        double mseThreshold = 0.00001;
        double learningRate = 0.1;

        double[][] trainingInputs = loadInputs("dataset.csv", 1, 500);
        double[] trainingTargets = loadTargets("targets.csv", 1, 500);

        if (trainingInputs.length != trainingTargets.length) {
            System.err.println("Error: The number of training inputs does not match the number of training targets.");
            return;
        }

        ArrayList<Layer> layers = new ArrayList<>();
        layers.add(new Layer(1, 400));

        NeuralNetwork nn = new NeuralNetwork(layers);

        long trainingStartTime = 0;
        long trainingEndTime = 0;
        double finalMse = Double.NaN;

        if (shouldTrain) {
            trainingStartTime = System.nanoTime();
            finalMse = nn.train(trainingInputs, trainingTargets, mseThreshold, learningRate);
            trainingEndTime = System.nanoTime();
        } else {
            nn.loadWeights("pesos.csv");
        }

        double[][] testInputs = loadInputs("dataset.csv", 501, 300);
        double[] testTargets = loadTargets("targets.csv", 501, 300);

        if (testInputs.length != testTargets.length) {
            System.err.println("Error: The number of test inputs does not match the number of test targets.");
            return;
        }

        double trainingTimeSeconds = (trainingEndTime - trainingStartTime) / 1_000_000_000.0;

        System.out.println("=================================================");
        System.out.println("                Testing the Network              ");
        System.out.println("=================================================");
        nn.test(testInputs, testTargets);

        System.out.println("TRAINING - TRAINED INPUTS: " + trainingTargets.length);
        System.out.println("TRAINING - MSE: " + mseThreshold);
        System.out.println("TRAINING - Learning Rate: " + learningRate);
        System.out.println("Final MSE after training: " + finalMse);
        System.out.println("Training Time (s): " + String.format("%.4f", trainingTimeSeconds));
        System.out.println("=================================================");
        System.out.println("            End of Testing Metrics               ");
        System.out.println("=================================================");
    }

    /**
     * Executes the "mooshake" operation by loading weights and processing user-provided inputs.
     */
    public void mooshake() {
        Scanner scanner = new Scanner(System.in);

        ArrayList<Layer> layers = new ArrayList<>();
        layers.add(new Layer(16, 400));
        layers.add(new Layer(16, 16));
        layers.add(new Layer(16, 16));
        layers.add(new Layer(16, 1));

        NeuralNetwork nn = new NeuralNetwork(layers);
        nn.loadWeights("pesos.csv");

        double[][] testInputs = readInputsFromConsole(scanner);
        nn.mooshake(testInputs);
        scanner.close();
    }

    /**
     * Normalizes the input array to a range between 0 and 1.
     *
     * @param inputs The array of input values.
     * @return A new array with normalized values.
     */
    public double[] normalizeInputs(double[] inputs) {
        double min = Arrays.stream(inputs).min().orElse(0.0);
        double max = Arrays.stream(inputs).max().orElse(1.0);

        if (max - min == 0) {
            return Arrays.stream(inputs).map(v -> 0.0).toArray();
        }

        return Arrays.stream(inputs)
                .map(v -> (v - min) / (max - min))
                .toArray();
    }

    /**
     * Reads input values from the console, normalizes them, and returns them as a 2D array.
     *
     * @param scanner The Scanner instance for reading console input.
     * @return A 2D array containing the normalized input values.
     */
    private double[][] readInputsFromConsole(Scanner scanner) {
        List<double[]> inputsList = new ArrayList<>();
        String line = scanner.nextLine();
        String[] tokens = line.split(",");
        double[] input = new double[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            input[i] = Double.parseDouble(tokens[i]);
        }
        input = normalizeInputs(input);
        inputsList.add(input);
        return inputsList.toArray(new double[0][]);
    }

    /**
     * Loads input data from a CSV file starting from a specific line and reads a maximum number of lines.
     *
     * @param filename   The path to the CSV file.
     * @param startLine  The line number to start reading from.
     * @param maxLines   The maximum number of lines to read.
     * @return A 2D array containing the input data.
     */
    public double[][] loadInputs(String filename, int startLine, int maxLines) {
        List<double[]> inputsList = new ArrayList<>();
        int linesRead = 0;
        int currentLine = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                currentLine++;
                if (currentLine < startLine) {
                    continue;
                }
                if (linesRead >= maxLines) {
                    break;
                }
                String[] tokens = line.split(",");
                if (tokens.length < 400) {
                    continue;
                }
                double[] input = new double[400];
                for (int i = 0; i < 400; i++) {
                    input[i] = Double.parseDouble(tokens[i]);
                }
                inputsList.add(input);
                linesRead++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return inputsList.toArray(new double[0][]);
    }

    /**
     * Loads target values from a CSV file starting from a specific line and reads a maximum number of lines.
     *
     * @param filename   The path to the CSV file.
     * @param startLine  The line number to start reading from.
     * @param maxLines   The maximum number of lines to read.
     * @return An array containing the target values.
     */
    public double[] loadTargets(String filename, int startLine, int maxLines) {
        List<Double> targetsList = new ArrayList<>();
        int linesRead = 0;
        int currentLine = 0;

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                currentLine++;
                if (currentLine < startLine) {
                    continue;
                }
                if (linesRead >= maxLines) {
                    break;
                }
                String[] tokens = line.split(",");
                if (tokens.length < 1) {
                    continue;
                }
                double target = Double.parseDouble(tokens[0]);
                targetsList.add(target);
                linesRead++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[] targets = new double[targetsList.size()];
        for (int i = 0; i < targetsList.size(); i++) {
            targets[i] = targetsList.get(i);
        }

        return targets;
    }
}