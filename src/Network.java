public class Network {
  private Matrix hiddenLayer;
  private Matrix outputLayer;
  private Matrix oldDeltaWHiddenLayer;
  private Matrix oldDeltaWOutputLayer;
  private Matrix hiddenLayerOutput;
  private Matrix outputLayerOutput;
  private Matrix outputLayerOutputErrors;
  private Matrix hiddenLayerOutputErrors;

  private double mse; //Mean Square Error

    public Network(int inputs, int outputs, int hiddenLayerNeurons, boolean isBias)
    {
        this.hiddenLayer = new Matrix(hiddenLayerNeurons, inputs+ (isBias?1:0), true);
        this.outputLayer = new Matrix(outputs, hiddenLayerNeurons+ (isBias?1:0), true);
        this.oldDeltaWHiddenLayer = new Matrix(hiddenLayerNeurons, inputs+ (isBias?1:0));
        this.oldDeltaWOutputLayer = new Matrix(outputs, hiddenLayerNeurons+ (isBias?1:0));
    }
    public void generateOutputs(double[] input, boolean isBias)
    {
        if(isBias) {
            double[] newInput = new double[input.length+1];
            newInput[0] = 1;
            for(int i=1;i<newInput.length;i++) newInput[i] = input[i-1];
            input = newInput;
        }

        double[] outputWithoutY = new double[input.length-3]; //todo powinno odkrajac input z wejscia, tesciki
        for(int i = 0; i<outputWithoutY.length;i++){
            outputWithoutY[i] = input[i];
        }

        hiddenLayerOutput = hiddenLayer.multiplication(outputWithoutY).sigmoid();
        outputLayerOutput = outputLayer.multiplication(isBias?hiddenLayerOutput.withBias():hiddenLayerOutput).sigmoid();

        countTotalMSE(input, isBias);
    }
    public void countTotalMSE(double[]input_row, boolean isBias)
    {
        double mse = 0;
        for(int i=0;i<outputLayerOutput.getMatrix().length;i++)
        {
            mse +=Math.pow((input_row[input_row.length-3+i]-outputLayerOutput.getMatrix()[i][0]),2);
        //    System.out.println("input_row: "+input_row[input_row.length-3+i]+", policzone: "+ String.format("%.10f", outputLayerOutput.getMatrix()[i][0])+", mse: "+ String.format("%.10f",Math.pow((input_row[input_row.length-3+i]-outputLayerOutput.getMatrix()[i][0]),2)));

        }
        //System.out.println();
        //System.out.println("input_row: "+input_row[(isBias?1:0)]+", policzone: "+ String.format("%.10f", outputLayerOutput.getMatrix()[0][0])+", mse: "+ String.format("%.10f",Math.pow((input_row[1+(isBias?1:0)]-outputLayerOutput.getMatrix()[0][0]),2)));
        this.mse = mse/2.0;
    }
    public double countMseForTest(double[] inputRow, boolean isBias)
    {
        if(isBias) {
            double[] newInput = new double[inputRow.length+1];
            newInput[0] = 1;
            for(int i=1;i<newInput.length;i++) newInput[i] = inputRow[i-1];
            inputRow = newInput;
        }
        double[] outputWithoutY = new double[inputRow.length-3];
        for(int i = 0; i<outputWithoutY.length;i++){
            outputWithoutY[i] = inputRow[i];
        }

        Matrix hidden = hiddenLayer.multiplication(outputWithoutY).sigmoid();
       Matrix output = outputLayer.multiplication(isBias?hidden.withBias():hidden).sigmoid();

        double mse = 0;
        for(int i=0;i<outputLayerOutput.getMatrix().length;i++)
        {
            mse +=Math.pow((inputRow[inputRow.length-3+i]-output.getMatrix()[i][0]),2);
        }
        return mse/2.0;

    }
    private void backPropagationForOutputLayer(double[] expectedOutput)
    {
        Matrix errors = new Matrix(outputLayerOutput.getMatrix().length,1);
        for(int i=0;i<outputLayerOutput.getMatrix().length;i++)
        {
            errors.getMatrix()[i][0] = (outputLayerOutput.getMatrix()[i][0] - expectedOutput[expectedOutput.length-3+i])*
                    outputLayerOutput.getMatrix()[i][0]*(1-outputLayerOutput.getMatrix()[i][0]);
        }
        outputLayerOutputErrors = errors;
    }
    private void backPropagationForHiddenLayer(boolean isBias)
    {
        Matrix errors = new Matrix(hiddenLayerOutput.getMatrix().length,1);
            for(int i=0;i<hiddenLayerOutput.getMatrix().length;i++)
            { //dla kazdego z neuronow hiddena
                //for (int j=0;j<outputLayer.getMatrix()[0].length;j++) <--???? chyba ok, ale raczej nie
                for (int j=0;j<outputLayer.getMatrix().length;j++)
                { //dla kazdej z wag w warstwie wyjsciowej
                   errors.getMatrix()[i][0] += outputLayerOutputErrors.getMatrix()[j][0]*outputLayer.getMatrix()[j][i+(isBias?1:0)];
                }
                errors.getMatrix()[i][0] *= hiddenLayerOutput.getMatrix()[i][0]*(1-hiddenLayerOutput.getMatrix()[i][0]);
            }
            hiddenLayerOutputErrors = errors;
    }
    public void backPropagation(double[]input, boolean isBias)
    {
        backPropagationForOutputLayer(input);
        backPropagationForHiddenLayer(isBias);
    }
    public void updateWeights(double[] input, double learningRate, double momentumRate, boolean isBias)
    {
        Matrix deltaWOutputLayer = new Matrix(outputLayer.getMatrix().length, outputLayer.getMatrix()[0].length);
        Matrix deltaWHiddenLayer = new Matrix(hiddenLayer.getMatrix().length, hiddenLayer.getMatrix()[0].length);
        if(isBias) {
            double[] newInput = new double[input.length+1];
            newInput[0] = 1;
            for(int i=1;i<newInput.length;i++) newInput[i] = input[i-1];
            input = newInput;
        }
        for(int i=0;i<outputLayer.getMatrix().length;i++)
        {//operacja dla kazdego z neuronów
            for(int j=0;j<outputLayer.getMatrix()[0].length;j++)
            {//dla j-tej wagi w i-tym neuronie
                deltaWOutputLayer.getMatrix()[i][j] =
                        -1 * learningRate* outputLayerOutputErrors.getMatrix()[i][0] * (isBias?hiddenLayerOutput.withBias() : hiddenLayerOutput).getMatrix()[j][0] +
                                momentumRate *oldDeltaWOutputLayer.getMatrix()[i][j];
            }
        }
        //i dla hiddenu:
        for(int i=0;i<hiddenLayer.getMatrix().length;i++)
        {//operacja dla kazdego z neuronów
            for(int j=0;j<hiddenLayer.getMatrix()[0].length;j++)
            {//dla j-tej wagi w i-tym neuronie
                deltaWHiddenLayer.getMatrix()[i][j] =
                        -1 * learningRate* hiddenLayerOutputErrors.getMatrix()[i][0] * input[j] +//hiddenLayerOutput.getMatrix()[j][0] +
                                momentumRate *oldDeltaWHiddenLayer.getMatrix()[i][j];
            }
        }
        hiddenLayer.add(deltaWHiddenLayer);
        outputLayer.add(deltaWOutputLayer);

        oldDeltaWHiddenLayer = new Matrix(deltaWHiddenLayer.getMatrix());
        oldDeltaWOutputLayer = new Matrix(deltaWOutputLayer.getMatrix());
    }
    public double getTotalMSE()
    {
        return mse;
    }
    public Matrix getHiddenLayer() {
        return hiddenLayer;
    }
    public Matrix getOutputLayer() {
        return outputLayer;
    }
    public Matrix getOldDeltaWHiddenLayer() {
        return oldDeltaWHiddenLayer;
    }
    public Matrix getOldDeltaWOutputLayer() {
        return oldDeltaWOutputLayer;
    }
    public Matrix getHiddenLayerOutput() {
        return hiddenLayerOutput;
    }
    public Matrix getOutputLayerOutput() {
        return outputLayerOutput;
    }
    //((to co wyszlo) -( to co mialo wyjsc))*(funkcja aktywacji(wartosc_na_wyjsciu))*(1 - funkcja aktywacji(wartosc_na_wyjsciu))

}
