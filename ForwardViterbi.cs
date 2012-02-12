/*
 * ForwardViterbi: Online Viterbi Algorithm
 * Edited by Zang
 */
//081220


using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using Globalspace;


namespace ReadRate
{
    public class ForwardViterbi
    {

        // The Class Globals
        string[] states;
        string[] observations;
        double[] startProbability;
        double[,] transitionProbability;
        double[,] emissionProbability;
        Histo histStill;
        Histo histMoving;
        double scaleFactor;


        int maxTimeUnit = 180;
        //Computed Variables
        int[] finalPath;
        int[,] vPath; //The Viterbi Path
        int currob = 0;
        int prevob = -1;
        int currob_state = 0;

        double[,] problem;
        int[,] problemfile;
        public int pausetime = 100;

        public double ErrorRate;
        public int Errors;
        public double Delay;
        public double AverDelay;
        public int DataSize;

        private Form1 frm1;

        /// <summary>
        ///the thread starts
        /// </summary>
        public event EventHandler threadStartEvent;
        /// <summary>
        /// handler of AddMotion event
        /// </summary>
        public event EventHandler threadEvent_AddMotion;
        /// <summary>
        /// handler of AddValue event
        /// </summary>
        public event EventHandler threadEvent_AddValue;
        public event EventHandler threadEvent_AddReadRate;

        /// <summary>
        /// event of thread ends
        /// </summary>
        public event EventHandler threadEndEvent;



        //----------------------------------------------------------------------
        // The Getters or Accessors

        public int[,] VPath
        {
            get { return vPath; }
        }


        //----------------------------------------------------------------------
        //Constructor
        public ForwardViterbi(Form1 frm, string[] states, string[] observations, double[] startProbability, double[,] transitionProbability, double[,] emissionProbability, double scaleFactor)
        {
            frm1 = frm;
            this.states = states;
            this.observations = observations;
            this.startProbability = startProbability;
            this.transitionProbability = transitionProbability;
            this.emissionProbability = emissionProbability;
            this.scaleFactor = scaleFactor;

        }

        public ForwardViterbi(Form1 frm, string[] states, string[] observations, double[] startProbability, double[,] transitionProbability, Histo hstill, Histo hmoving, double scaleFactor)
        {
            frm1 = frm;
            this.states = states;
            this.observations = observations;
            this.startProbability = startProbability;
            this.transitionProbability = transitionProbability;
            this.histMoving = hmoving;
            this.histStill = hstill;
            this.scaleFactor = scaleFactor;
        }


        public void setProblem(int[,] prob)
        {
            this.problemfile = prob;
            this.problem = getVarSequence(prob);
        }

        public void setPT(int v)
        {
            pausetime = v;
        }

        private double[,] getVarSequence(int[,] data)
        {
            //we won't use the last ramaining data ; 
            int datanum = data.GetLength(0);
            int[] answer = new int[global.WINDOWLENTH];
            int len = datanum / global.WINDOWLENTH;
            double[,] ans = new double[len, 2];
            int i = 0;
            int numberofone = 0;
            int winnum = 0;
            int s = 0;
            for (int count = 0; count < datanum; count++)
            {
                answer[i] = data[count, 0];
                if (data[count, 1] == 1) numberofone++;
                i++;
                if (i >= global.WINDOWLENTH)
                {
                    ans[winnum, 0] = GetVariance(answer);
                    ans[winnum, 1] = Convert.ToDouble((numberofone / global.WINDOWLENTH > 0.5));
                    if (ans[winnum, 1] == 0)
                        s = 1;
                    winnum++;
                    i = 0;
                    numberofone = 0;
                }

            }
            return ans;
        }

        //------------------------------------------------
        private double GetAverage(int[] array)
        {
            double average = 0;
            if (array.Length > 0)
            {
                double total = 0;
                for (int i = 0; i < array.Length; i++)
                {
                    total += array[i];
                }
                average = total / array.Length;
            }
            return average;
        }
        private double GetVariance(int[] array)
        {
            double average = GetAverage(array);
            double fto = 0;
            for (int i = 0; i < array.Length; i++)
            {
                fto += Math.Pow((average - array[i]), 2);
            }
            double u2 = Math.Pow(average, 2);
            double equation = fto / u2;
            return equation * 10;
        }


        /*
         OLP Process
         */

        /*
         * the online HMM process using histogram
         */
        public void onlineProcess()
        {
            //threadStartEvent.Invoke(count, new EventArgs());//notify the mainthread
            int datanum = problem.GetLength(0);
            DataSize = datanum;
            double[] x = new double[datanum];
            for (int i = 0; i < datanum; i++)
            {
                x[i] = problem[i, 1];
            }

            double[,] T = new double[states.Length, 3];  //We will store the probability sequence for the Viterbi Path
            finalPath = new int[datanum];
            vPath = new int[states.Length, datanum];
            for (int state = 0; state < states.Length; state++)
            {
                vPath[state, 0] = state;
            }

            //initialize T
            //------------------------------------------------------------------	
            for (int state = 0; state < states.Length; state++)
            {
                T[state, 0] = Math.Log(startProbability[state]);
                T[state, 1] = state;
                T[state, 2] = Math.Log(startProbability[state]);
            }

            for (int output = 0; output < datanum - 1; output++)
            {
                double[,] U = new double[states.Length, 3];  //We will use this array to calculate the future probabilities
                
                Thread.Sleep(pausetime);

                int[] Param = new int[global.WINDOWLENTH];
                for (int j = 0; j < global.WINDOWLENTH; j++)
                {
                    Param[j] = problemfile[global.WINDOWLENTH * output + j, 0];
                }
                threadEvent_AddReadRate.Invoke(Param, new EventArgs());
                threadEvent_AddValue.Invoke(problem[output, 0], new EventArgs());

                string pp ;
                for (int nextState = 0; nextState < states.Length; nextState++)
                {
                    double total = 0;
                    int argMax = 0;
                    double valMax = -32768;

                    for (int state = 0; state < states.Length; state++)
                    {

                        Console.WriteLine("    The testing state is {0} ({1})", states[state], state);
                        double v_prob = T[state, 0];
                        double v_path = T[state, 1];

                        double emissionProb = 1;
                        if (state == 0)
                        {
                            emissionProb = histStill.Prob(problem[output, 0]);
                        }
                        else
                        {
                            emissionProb = histMoving.Prob(problem[output, 0]);
                        }
                        if (emissionProb == 0.0)
                            emissionProb = 0.001;


                        double p = Math.Log(emissionProb) + Math.Log(transitionProbability[state, nextState]) ;
                        v_prob += p;


                        if (v_prob > valMax)
                        {
                            valMax = v_prob;
                            argMax = state;
                        }
                    }

                    U[nextState, 0] = valMax;
                    U[nextState, 1] = argMax;
                    U[nextState, 2] = valMax;

                    vPath[nextState, output + 1] = argMax;

                }

                T = U;

                //determine if the fusion point shows up
                if (T[0, 1] == T[1, 1])
                {
                    //ouput the segment from the last fusion point
                    currob = output;
                    currob_state = vPath[0, output + 1]; //T[0, 1] = T[1, 1] = vPath[0, output + 1] = vPath[1, output + 1] 
                    updateOutput();
                    prevob = currob;

                }
                else
                {
                }

                currob++;

            }

            //remaining output
            double Total = 0;
            double ValMax = -32768;
            double ArgMax = 0;

            
            int finalobs = datanum - 1;  //this is the final observation

            for (int state = 0; state < states.Length; state++)
            {
                double prob = T[state, 0];
                double v_path = T[state, 1];

                double emissionProb = 1;
                if (state == 0)
                {
                    emissionProb = histStill.Prob(problem[finalobs, 0]);
                }
                else
                {
                    emissionProb = histMoving.Prob(problem[finalobs, 0]);
                }

                double v_prob = T[state, 2] + Math.Log(emissionProb);

                if (v_prob > ValMax)
                {
                    ValMax = v_prob;
                    ArgMax = state;
                }
            }

            currob = datanum - 1;
            updateOutput();

            int rights = 0;
            for (int cnt = 0; cnt < datanum - 1; cnt++)
            {
                if ((double)(finalPath[cnt]) == problem[cnt, 1])
                    rights++;
            }

            Errors = datanum - rights;
            ErrorRate = (double)rights / (double)datanum;
            ErrorRate = 1 - ErrorRate;
            AverDelay = Delay / (double)datanum;

            threadEndEvent.Invoke(0, new EventArgs());
       }


        //output the segment from the previsou fusion point to the current fusion point
        private void updateOutput()
        {
            int nowstate = currob_state ;
            int[] param = new int[currob - prevob];

            for(int i = currob; i > prevob; i--)
            {
                finalPath[i] = nowstate ;
                param[i-prevob-1] = nowstate ;
                nowstate = vPath[nowstate, i];
                Delay += currob - i;
                
            }
            threadEvent_AddMotion.Invoke(param, new EventArgs());//notify the main thread
        }

    }; // end Forward Viterbi Class
}
