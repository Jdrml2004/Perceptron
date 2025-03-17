import java.util.ArrayList;

/**
 * Represents a layer in the neural network, consisting of multiple neurons.
 */
public class Layer {
    private ArrayList<Neuron> neurons;

    /**
     * Constructs a Layer with a specified number of neurons, each initialized with a given input size.
     *
     * @param numNeurons The number of neurons in the layer.
     * @param inputSize  The number of inputs each neuron receives.
     */
    public Layer(int numNeurons, int inputSize) {
        neurons = new ArrayList<>();
        for (int i = 0; i < numNeurons; i++) {
            ArrayList<Double> weights = new ArrayList<>();
            for (int j = 0; j < inputSize; j++) {
                weights.add(Math.random() * 0.01);
            }
            neurons.add(new Neuron(weights, 0.0));
        }
    }

    /**
     * Performs a forward pass through the layer by activating each neuron with the provided inputs.
     *
     * @param inputs The input values to the layer.
     * @return An array of output values from the layer.
     */
    public double[] forward(double[] inputs) {
        double[] outputs = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            outputs[i] = neurons.get(i).activate(inputs);
        }
        return outputs;
    }

    /**
     * Retrieves the list of neurons in the layer.
     *
     * @return An ArrayList of neurons.
     */
    public ArrayList<Neuron> getNeurons() {
        return neurons;
    }
}