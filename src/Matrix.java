import java.util.Random;

public class Matrix implements Cloneable {
    private double[][] matrix;

    public Matrix(int row, int col) {
        this.matrix = new double[row][col];
    }
    public Matrix(int row, int col, boolean isRandom) {
        this.matrix = new double[row][col];
        if(isRandom)
        {
            Random r = new Random();
            for(int i=0;i<this.matrix.length;i++)
            {
                for(int j=0;j<this.matrix[0].length;j++)
                {

                    this.matrix[i][j] = r.nextDouble()*2-1;
                }
            }
        }
    }

    public Matrix(double[][] m) {
        this.matrix = m;
    }
   public void add(Matrix addend)
    {
        if((this.matrix.length != addend.matrix.length)||
                (this.matrix[0].length != addend.matrix[0].length)) throw new IndexOutOfBoundsException();
        for(int i=0;i<matrix.length;i++)
        {
            for(int j=0;j<matrix[0].length;j++)
            {
                this.matrix[i][j]+=addend.matrix[i][j];
            }
        }
    }
    public void subtract(Matrix subtrahend)
    {
        if((this.matrix.length != subtrahend.matrix.length)||
                (this.matrix[0].length != subtrahend.matrix[0].length)) throw new IndexOutOfBoundsException();
        for(int i=0;i<matrix.length;i++)
        {
            for(int j=0;j<matrix[0].length;j++)
            {
                this.matrix[i][j]-=subtrahend.matrix[i][j];
            }
        }
    }
   public Matrix multiplication(Matrix multi)
   {
       if(this.matrix[0].length != multi.matrix.length) throw new IndexOutOfBoundsException();
       if(multi.matrix[0].length!=1) throw new IndexOutOfBoundsException("125");
       Matrix product = new Matrix(this.matrix.length,multi.matrix[0].length);

       for(int i =0;i<this.matrix.length;i++) {
           for(int j =0;j<multi.matrix[0].length;j++) {

               //ponizsze wykonywane dla kazdego elementu docelowej tablicy
               for(int k=0;k<this.matrix[0].length; k++) {
                   product.matrix[i][j] += (this.matrix[i][k] * multi.matrix[k][j]);
               }
           }
       }
       return product;
   }

   public Matrix multiplication(double[] m)
   {
       return multiplication(transpose(m));
   }
    public Matrix multiplication(double x)
    {
        /*Matrix output = new Matrix(matrix.length,matrix[0].length) ;
        for(int i =0;i<matrix.length;i++) {
            for(int j =0;j<matrix[0].length;j++) {
                    output.matrix[i][j]= this.matrix[i][j]*x;
            }
        }
        return output;*/
        Matrix output = new Matrix(matrix.length,1) ;
        for(int i =0;i<matrix.length;i++) {
            for(int j =0;j<matrix[0].length;j++) {
                output.matrix[i][0]+= this.matrix[i][j]*x;
            }
        }
        return output;
    }
   @Override
   public String toString() {
        StringBuilder s = new StringBuilder();
        for (int i=0;i<this.matrix.length;i++) {
            for (int j = 0; j<this.matrix[0].length; j++) {
                //s.append(this.matrix[i][j]);
                s.append(String.format("%.10f", this.matrix[i][j]));
                s.append("\t");
            }
            s.append("\n");
        }
       return s.toString();
   }
   public Matrix withBias()
   {
       Matrix columnWithBias = new Matrix(this.matrix.length+1,1);
       columnWithBias.matrix[0][0]=1;
       for(int i=1;i<this.matrix.length+1;i++)
       {
           columnWithBias.matrix[i][0]=this.matrix[i-1][0];
       }
       return columnWithBias;
   }

    public Matrix sigmoid()
    {
        Matrix output = new Matrix(this.matrix.length,1);
        for(int i=0;i<this.matrix.length;i++)
        {
            output.matrix[i][0] = (1/(1+Math.exp(-this.matrix[i][0])));
        }
        return output;
    }
    public Matrix transpose(double[] m)
    {
        Matrix output = new Matrix(m.length,1);
        for(int i=0;i<m.length;i++)
        {
            output.matrix[i][0]=m[i];
        }
        return  output;
    }

    public double[][] getMatrix() {
        return matrix;
    }

}

