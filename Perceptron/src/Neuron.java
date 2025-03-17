import java.util.ArrayList;

/**
 * Represents a single neuron within a neural network layer, handling input processing and activation.
 */
public class Neuron {

    private ArrayList<Double> weights;
    private double bias;
    private double output;
    private double delta;

    /**
     * Constructs a Neuron with specified weights and bias.
     *
     * @param weights The weights associated with the neuron's inputs.
     * @param bias    The bias value for the neuron.
     */
    public Neuron(ArrayList<Double> weights, double bias) {
        this.weights = weights;
        this.bias = bias;
    }

    /**
     * Calculates the net input (weighted sum plus bias) for the neuron based on the provided inputs.
     *
     * @param inputs The input values to the neuron.
     * @return The net input value.
     */
    public double netInput(double[] inputs) {
        double sum = bias;
        for (int i = 0; i < weights.size(); i++) {
            sum += weights.get(i) * inputs[i];
        }
        return sum;
    }

    /**
     * Activates the neuron using the sigmoid activation function based on the provided inputs.
     *
     * @param inputs The input values to the neuron.
     * @return The activated output value.
     */
    public double activate(double[] inputs) {
        double netInput = netInput(inputs);
        output = sigmoid(netInput);
        return output;
    }

    /**
     * Applies the sigmoid activation function to a given value.
     *
     * @param value The value to be transformed.
     * @return The sigmoid of the input value.
     */
    private double sigmoid(double value) {
        return 1 / (1 + Math.exp(-value));
    }

    /**
     * Calculates the derivative of the sigmoid function based on the neuron's current output.
     *
     * @return The derivative of the sigmoid function.
     */
    public double sigmoidDerivative() {
        return output * (1 - output);
    }

    /**
     * Retrieves the list of weights associated with the neuron.
     *
     * @return An ArrayList of weight values.
     */
    public ArrayList<Double> getWeights() {
        return weights;
    }

    /**
     * Sets the weights for the neuron.
     *
     * @param weights An ArrayList of new weight values.
     */
    public void setWeights(ArrayList<Double> weights) {
        this.weights = weights;
    }

    /**
     * Retrieves the bias value of the neuron.
     *
     * @return The bias value.
     */
    public double getBias() {
        return bias;
    }

    /**
     * Sets the bias value of the neuron.
     *
     * @param bias The new bias value.
     */
    public void setBias(double bias) {
        this.bias = bias;
    }

    /**
     * Retrieves the last output value produced by the neuron.
     *
     * @return The output value.
     */
    public double getOutput() {
        return output;
    }

    /**
     * Retrieves the delta value used for weight updates during backpropagation.
     *
     * @return The delta value.
     */
    public double getDelta() {
        return delta;
    }

    /**
     * Sets the delta value for the neuron.
     *
     * @param delta The new delta value.
     */
    public void setDelta(double delta) {
        this.delta = delta;
    }
}