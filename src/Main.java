import com.panayotis.gnuplot.JavaPlot;
import com.panayotis.gnuplot.plot.DataSetPlot;
import com.panayotis.gnuplot.style.NamedPlotColor;
import com.panayotis.gnuplot.style.PlotStyle;
import com.panayotis.gnuplot.style.Style;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws Exception {
        JavaPlot jp = new JavaPlot("C:\\gnuplot\\bin\\gnuplot.exe");
        String contents = new String(Files.readAllBytes(Paths.get("classification_train.txt")));
        LinkedList<Double> inputValue = new LinkedList<>();
        LinkedList<Double> inputValueTest = new LinkedList<>();
        String inputTest = new String(Files.readAllBytes(Paths.get("classification_test.txt")));
        Scanner sc = new Scanner(contents.replace('.',','));
        Scanner sc2 = new Scanner(inputTest.replace('.',','));
        while(sc.hasNext()) {
            inputValue.add(sc.nextDouble());
        }
        while(sc2.hasNext()) {
            inputValueTest.add(sc2.nextDouble());
        }
        double[][] input_train=new double[inputValue.size()/5][4+3];
        double[][] input_test=new double[inputValueTest.size()/5][4+3];

        for (int i=0;i<inputValue.size()/5;i++)
        {
            for(int j=0;j<4;j++)
            input_train[i][j]=inputValue.get(i*5+j);
            input_train[i][4] = (inputValue.get(i*5+4)==1?1:0);
            input_train[i][5] = (inputValue.get(i*5+4)==2?1:0);
            input_train[i][6] = (inputValue.get(i*5+4)==3?1:0);
        }
        for (int i=0;i<inputValueTest.size()/5;i++)
        {
            for(int j=0;j<4;j++)
            input_test[i][j]=inputValueTest.get(i*5+j);
            input_test[i][4] = (inputValueTest.get(i*5+4)==1?1:0);
            input_test[i][5] = (inputValueTest.get(i*5+4)==2?1:0);
            input_test[i][6] = (inputValueTest.get(i*5+4)==3?1:0);
        }


        /* EVIDENCE GATHERING
        for (int i = 0; i< input_test.length; i++) {
            if(i%3==0) {
                for(double j : input_test[i])
                System.out.print(j + "\t");
                System.out.println();

            }
        }
        System.out.println("\n");
        for (int i = 0; i< input_test.length; i++) {
            if(i%3==1) {
                for(double j : input_test[i])
                    System.out.print(j + "\t");
                System.out.println();
            }
            if(i==43) {
                System.out.println("^^^");
            }
        }
        System.out.println("\n");
        for (int i = 0; i< input_test.length; i++) {
            if(i%3==2) {
                for(double j : input_test[i])
                    System.out.print(j + "\t");
                System.out.println();

            }
        }
        System.out.println("\n");
        */

        /*String check = new String(Files.readAllBytes(Paths.get("approximation_train_2.txt")));
        Scanner sc2 = new Scanner(check.replace('.',','));
        LinkedList<Double> inputValueCheck = new LinkedList<>();
        LinkedList<Double> expectedValueCheck = new LinkedList<>();
        while(sc2.hasNext()) {
            inputValueCheck.add(sc2.nextDouble());
            expectedValueCheck.add(sc2.nextDouble());
        }
        double[][] input_train_15pts=new double[inputValueCheck.size()][2];
        for (int i=0;i<inputValueCheck.size();i++)
        {
            input_train_15pts[i][0]=inputValueCheck.get(i);
            input_train_15pts[i][1]=expectedValueCheck.get(i);
        }

        String test = new String(Files.readAllBytes(Paths.get("approximation_test.txt")));
        Scanner sc3 = new Scanner(test.replace('.',','));
        LinkedList<Double> inputValueTest = new LinkedList<>();
        LinkedList<Double> expectedValueTest = new LinkedList<>();
        while(sc3.hasNext()) {
            inputValueTest.add(sc3.nextDouble());
            expectedValueTest.add(sc3.nextDouble());
        }
        double[][] input_test_1000pts=new double[inputValueTest.size()][2];
        for (int i=0;i<inputValueTest.size();i++)
        {
            input_test_1000pts[i][0]=inputValueTest.get(i);
            input_test_1000pts[i][1]=expectedValueTest.get(i);
        }*/

        /////////////////////////////////////////////////////
        //tresowanie inputu

        //wsadź tu indeksy do usuwania
        //int[] to_delete = new int[] {4, 3, 2, 1}; <-- in this order
        int[] to_delete = new int[] {};

        //System.out.println(new Matrix(input_train));
        //don't touch this i'm watching you
        for(int i : to_delete) {
            if (i == 4) {input_train = clipCol(input_train, 3);
                input_test = clipCol(input_test, 3);
            }
            if (i == 3) {input_train = clipCol(input_train, 2);
                input_test = clipCol(input_test, 2);
            }
            if (i == 2) {input_train = clipCol(input_train, 1);
                input_test = clipCol(input_test, 1);
            }
            if (i == 1) {input_train = clipCol(input_train, 0);
                input_test = clipCol(input_test, 0);
            }
        }

        /////////////////////////////////////////////////////
        int INPUTS = 4 - to_delete.length;
        int OUTPUTS= 3;
        int HIDDENLAYERNEURONS = 20;
        double LEARNINGRATE = 0.01;
        double MOMENTRATE = 0.5;
        boolean ISBIAS = true;
        int EPOCHLIMIT = 2000;
        double[][] TRAININGINPUTS = input_train;

        Network network = new Network(INPUTS, OUTPUTS, HIDDENLAYERNEURONS,ISBIAS);
        LinkedList<Integer> indeksy = new LinkedList<>();

        for(int i=0;i<TRAININGINPUTS.length;i++) {
            indeksy.add(i);
        }
        //Kolejne epoki
        LinkedList<Double> errors = new LinkedList<>();
        LinkedList<Double> errorsCheck = new LinkedList<>();
        //turn on mocked data VV
        int epochCounter = 1;
        double msefromEpoch;
        double mseFromCheck;
        do{
            Collections.shuffle(indeksy);
            msefromEpoch=0;
            mseFromCheck=0;
            for(int i: indeksy)
            {
                network.generateOutputs(TRAININGINPUTS[i],ISBIAS);
                network.backPropagation(TRAININGINPUTS[i], ISBIAS);
                network.updateWeights(TRAININGINPUTS[i], LEARNINGRATE, MOMENTRATE, ISBIAS);
                msefromEpoch+= network.getTotalMSE();
            }
            double compareMse = 0;

            for(int i=0;i<input_test.length - to_delete.length;i++)
                compareMse += network.countMseForTest(input_test[i],ISBIAS);

            //  System.out.println("Blad sieci po zbiorze testowym: "+compareMse);
            msefromEpoch/= OUTPUTS;
            msefromEpoch/=indeksy.size();
            errors.add(msefromEpoch);
            errorsCheck.add(compareMse/(input_test.length - to_delete.length));
            System.out.println("Epoka: "+ epochCounter + " MSE:   " + String.format("%.10f", msefromEpoch));
            epochCounter++;
        }while (epochCounter<EPOCHLIMIT );

        /////////////////////////////////////////////////////
        /* instrukcja do classes

        kolumny - podzielone na oczekiwane klasy - pierwsza zawiera elementy klasy 1 na wejsciu
        rzedy - podzielone na otrzymane klasy - pierwszy rzad zawiera elementy rozpoznane jako klasa 1
        */
        //zbierzemy wartości, sprawdzimy co umie
        int[][] classes={{0,0,0}, {0,0,0}, {0,0,0}};
        int expected_class = 0;
        for(double[] i : input_test) {
            for(int iterator = 0; iterator < 3;iterator++) {
                if(i[iterator+INPUTS] == 1) expected_class = iterator; //planujemy otrzymac indeks oczekiwanej sieci
            }
            network.generateOutputs(i, ISBIAS);
            double[] outputs = new double[3];
            for(int j = 0;j<OUTPUTS;j++){
                outputs[j] = network.getOutputLayerOutput().getMatrix()[j][0];
            }
            classes[classify(outputs)][expected_class]++;
        }
        System.out.println();
        for(int[] row : classes){
            for(int col : row) {
                System.out.print(col + " ");
            }
            System.out.println();
        }

        /////////////////////////////////////////////////////

        double [][]plotMse = new double[epochCounter-1][2];
        double [][]plotMseCompare=new double[epochCounter-1][2];

        for(int i=0;i<epochCounter-1;i++) {
            plotMse[i][0] = plotMseCompare[i][0] = i;
            plotMse[i][1] = errors.get(i);
            plotMseCompare[i][1]=errorsCheck.get(i);
        }

        DataSetPlot dsMse = new DataSetPlot(plotMse);
        dsMse.setTitle("MSE training");
        DataSetPlot dsMseCompare = new DataSetPlot(plotMseCompare);
        dsMseCompare.setTitle("MSE test");
        PlotStyle style = new PlotStyle();
        PlotStyle stylec = new PlotStyle();
        style.setStyle(Style.LINESPOINTS);
        stylec.setStyle(Style.LINESPOINTS);
        style.setPointType(7);
        stylec.setPointType(7);
        style.setLineType(NamedPlotColor.BLUE);
        stylec.setLineType(NamedPlotColor.DARK_SALMON);
        dsMseCompare.setPlotStyle(stylec);
        dsMse.setPlotStyle(style);

        jp.addPlot(dsMse);
        jp.addPlot(dsMseCompare);
        jp.plot();
    }
    private static double[][] clipCol(double[][] input, int index) {
        double[][] ret = new double[input.length][input[0].length-1];
        int index_offset = 0;
        for(int col_index = 0;col_index < ret[0].length;col_index++) {
            if(index == col_index) index_offset++;
                for(int row_index = 0; row_index < ret.length;row_index++) {
                ret[row_index][col_index] = input[row_index][col_index+index_offset];
            }
        }
        return ret;
    }
    private static int classify(double[] computed_values) {
        int index = 0;
        double max = computed_values[0];
        for(int i=1;i<3;i++) {
            if(computed_values[i]>max) {
                max = computed_values[i];
                index = i;
            }
        }
        return index;
    }
}
