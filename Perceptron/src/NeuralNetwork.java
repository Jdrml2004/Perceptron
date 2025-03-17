import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Represents a neural network composed of multiple layers and provides methods for training,
 * testing, and weight management.
 */
public class NeuralNetwork {

    private ArrayList<Layer> layers;
    private double learningRate;

    /**
     * Constructs the NeuralNetwork with the specified layers.
     *
     * @param layers The layers that make up the neural network.
     */
    public NeuralNetwork(ArrayList<Layer> layers) {
        this.layers = layers;
    }

    /**
     * Trains the neural network using the provided inputs and targets until the mean squared error
     * falls below the specified threshold or other stopping conditions are met.
     *
     * @param inputs       The training input data.
     * @param targets      The expected output values for the training data.
     * @param mseThreshold The threshold for the mean squared error to determine convergence.
     * @param learningRate The rate at which the network learns during training.
     * @return The final mean squared error after training.
     */
    public double train(double[][] inputs, double[] targets, double mseThreshold, double learningRate) {
        this.learningRate = learningRate;
        int epoch = 0;
        List<Double> mseHistory = new ArrayList<>();
        double finalMse = Double.NaN;

        try (BufferedWriter writer = new BufferedWriter(new FileWriter("mse_values.txt"))) {
            while (true) {
                double totalError = 0.0;

                for (int i = 0; i < inputs.length; i++) {
                    double[] input = inputs[i];
                    double target = targets[i];

                    double output = forward(input);
                    double error = target - output;
                    totalError += 0.5 * Math.pow(error, 2);

                    backward(target);
                    updateWeights(input);
                }

                double mse = totalError / inputs.length;
                mseHistory.add(mse);

                writer.write(String.format("%.100f", mse).replace('.', ','));
                writer.newLine();

                epoch++;

                if (mse < mseThreshold) {
                    finalMse = mse;
                    break;
                }

                if (epoch % 10 == 0 && epoch >= 10) {
                    double previousMSE = mseHistory.get(mseHistory.size() - 10);
                    if (mse > previousMSE) {
                        finalMse = mse;
                        break;
                    }
                }
            }

            saveWeights("pesos.csv");

        } catch (IOException e) {
            System.err.println("Error writing MSE values to file: " + e.getMessage());
        }

        return finalMse;
    }

    /**
     * Performs a forward pass through the entire network to compute the output.
     *
     * @param input The input data for the network.
     * @return The output value from the network.
     */
    public double forward(double[] input) {
        double[] outputs = input;
        for (Layer layer : layers) {
            outputs = layer.forward(outputs);
        }
        return outputs[0];
    }

    /**
     * Performs the backward pass to compute deltas for each neuron based on the target output.
     *
     * @param target The expected output value.
     */
    public void backward(double target) {
        Layer outputLayer = layers.get(layers.size() - 1);
        Neuron outputNeuron = outputLayer.getNeurons().get(0);
        double output = outputNeuron.getOutput();
        double delta = (target - output) * outputNeuron.sigmoidDerivative();
        outputNeuron.setDelta(delta);

        for (int i = layers.size() - 2; i >= 0; i--) {
            Layer currentLayer = layers.get(i);
            Layer nextLayer = layers.get(i + 1);
            for (int j = 0; j < currentLayer.getNeurons().size(); j++) {
                Neuron neuron = currentLayer.getNeurons().get(j);
                double sum = 0.0;
                for (Neuron nextNeuron : nextLayer.getNeurons()) {
                    sum += nextNeuron.getWeights().get(j) * nextNeuron.getDelta();
                }
                double neuronDelta = sum * neuron.sigmoidDerivative();
                neuron.setDelta(neuronDelta);
            }
        }
    }

    /**
     * Updates the weights and biases of all neurons in the network based on the computed deltas.
     *
     * @param input The original input data to the network.
     */
    public void updateWeights(double[] input) {
        double[] inputs = input;

        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                ArrayList<Double> weights = neuron.getWeights();
                for (int j = 0; j < weights.size(); j++) {
                    double oldWeight = weights.get(j);
                    double inputVal = inputs[j];
                    double newWeight = oldWeight + learningRate * neuron.getDelta() * inputVal;
                    weights.set(j, newWeight);
                }
                neuron.setBias(neuron.getBias() + learningRate * neuron.getDelta());
            }
            inputs = new double[layer.getNeurons().size()];
            for (int i = 0; i < layer.getNeurons().size(); i++) {
                inputs[i] = layer.getNeurons().get(i).getOutput();
            }
        }
    }

    /**
     * Tests the neural network with the provided test inputs and compares the outputs to the expected targets.
     *
     * @param inputs  The test input data.
     * @param targets The expected output values for the test data.
     */
    public void test(double[][] inputs, double[] targets) {
        System.out.println("Index | Expected | Predicted | Correct?");
        System.out.println("------+----------+-----------+---------");

        int correct = 0;
        double totalSquaredError = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double target = targets[i];
            double output = forward(input);
            int predictedClass = (output >= 0.5 ? 1 : 0);
            int expectedClass = (int) target;

            boolean isCorrect = (predictedClass == expectedClass);
            if (isCorrect) correct++;

            double squaredError = Math.pow((target - output), 2);
            totalSquaredError += squaredError;

            System.out.printf("%10d | %10d | %10f | %s%n", (i + 1), expectedClass, output, (isCorrect ? "Yes" : "No"));
        }

        double rmse = Math.sqrt(totalSquaredError / inputs.length);
        double accuracy = (double) correct / inputs.length * 100.0;

        System.out.println("=================================================");
        System.out.println("                TEST  RESULTS                    ");
        System.out.println("=================================================");
        System.out.println("Number of tests: " + inputs.length);
        System.out.println("Number of wrong guesses: " + (inputs.length - correct));
        System.out.printf("Accuracy: %.2f%%%n", accuracy);
        System.out.printf("RMSE: %.10f%n", rmse);
        System.out.println("=================================================");
        System.out.println("               TEST PARAMETERS                   ");
        System.out.println("=================================================");
    }

    /**
     * Executes the "mooshake" operation by processing the provided inputs and displaying the outputs.
     *
     * @param inputs The input data for the "mooshake" operation.
     */
    public void mooshake(double[][] inputs) {
        for (double[] input : inputs) {
            double output = forward(input);
            System.out.println(Math.round(output));
        }
    }

    /**
     * Predicts the output for a single input by performing a forward pass through the network.
     *
     * @param input The input data for prediction.
     * @return The predicted output value.
     */
    public double predict(double[] input) {
        return forward(input);
    }

    /**
     * Saves the current weights and biases of the neural network to a specified file.
     *
     * @param filename The name of the file to save the weights to.
     */
    public void saveWeights(String filename) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (Layer layer : layers) {
                for (Neuron neuron : layer.getNeurons()) {
                    for (double weight : neuron.getWeights()) {
                        writer.write(weight + ",");
                    }
                    writer.write(neuron.getBias() + "");
                    writer.newLine();
                }
            }
            System.out.println("Weights saved to " + filename);
        } catch (IOException e) {
            System.err.println("Error saving weights: " + e.getMessage());
        }
    }

    /**
     * Loads weights and biases for the neural network from a specified file.
     *
     * @param filename The name of the file to load the weights from.
     */
    public void loadWeights(String filename) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            int layerIndex = 0;
            int neuronIndex = 0;

            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(",");
                Layer layer = layers.get(layerIndex);
                Neuron neuron = layer.getNeurons().get(neuronIndex);

                ArrayList<Double> weights = new ArrayList<>();
                for (int i = 0; i < tokens.length - 1; i++) {
                    weights.add(Double.parseDouble(tokens[i]));
                }
                double bias = Double.parseDouble(tokens[tokens.length - 1]);

                neuron.setWeights(weights);
                neuron.setBias(bias);

                neuronIndex++;
                if (neuronIndex >= layer.getNeurons().size()) {
                    neuronIndex = 0;
                    layerIndex++;
                    if (layerIndex >= layers.size()) {
                        break;
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Erro ao carregar os pesos: " + e.getMessage());
        }
    }
}