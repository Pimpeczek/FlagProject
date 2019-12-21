using PiwotDrawingLib.UI;
using PiwotDrawingLib.Drawing;
using PiwotToolsLib.PMath;
using MathNet.Numerics;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;



namespace FlagProject
{
    class Program
    {
        static Vector<double>[] flagMatrixes;
        static int pWidth = 24;
        static int pHeight = 16;
        static int pTotal = pWidth * pHeight;
        static int memoryPoint = 0;
        static Int2 displaySize;
        static float[] memory;
        static List<float> memoryLong;
        static float maxErr = 0;
        static float minErr = float.MaxValue;
        static float maxErrLong = 1;
        static void Main(string[] args)
        {
            // Renderer.FrameLenght = 30;
            displaySize = new Int2(100, 10);
            memory = new float[displaySize.X-2];
            for (int i = 0; i < memory.Length; i++)
                memory[i] = 0;
            memoryLong = new List<float>();
            Renderer.WindowSize = new Int2(Arit.Clamp(Arit.Clamp(pWidth * 2, 12, int.MaxValue) + 9 + displaySize.X, 20, int.MaxValue),Arit.Clamp(pHeight+4 + pHeight%2, displaySize.Y*3, int.MaxValue));
            Renderer.AsyncMode = false;
            Renderer.WindowTitle = "FlagLearning";
            string[] filePaths = Directory.GetFiles("flags");
            Bitmap[] flagBitmaps = new Bitmap[filePaths.Length];
            flagMatrixes = new Vector<double>[filePaths.Length];
            PiwotDrawingLib.UI.Containers.PictureBox pb = new PiwotDrawingLib.UI.Containers.PictureBox(new Int2(1, 0), new Int2(pWidth+2, pHeight/2 + 2 + pHeight % 2), "OrgFlg", PiwotDrawingLib.Misc.Boxes.BoxType.round, new Bitmap(10, 10));
            PiwotDrawingLib.UI.Containers.PictureBox pb2 = new PiwotDrawingLib.UI.Containers.PictureBox(new Int2(pWidth + 5, 0), new Int2(pWidth + 2, pHeight / 2 + 2 + pHeight % 2), "GenFlg", PiwotDrawingLib.Misc.Boxes.BoxType.round, new Bitmap(10, 10));
            pb.SizeDifferenceHandling = PiwotDrawingLib.UI.Containers.Container.ContentHandling.CropContent;
            pb2.SizeDifferenceHandling = PiwotDrawingLib.UI.Containers.Container.ContentHandling.CropContent;
            pb.Draw();
            pb2.Draw();
            for (int i = 0; i < filePaths.Length; i++)
            {
                flagBitmaps[i] = PiwotToolsLib.PGraphics.Bitmaper.CutColorBits(PiwotToolsLib.PGraphics.Bitmaper.StreachToSize(new Bitmap(filePaths[i]), pWidth, pHeight), PiwotToolsLib.PGraphics.Coloring.ColorEncoding.Bit9);
                flagMatrixes[i] = Vector<double>.Build.Dense(pTotal*3, (x) =>
                {
                    int xpoint = (x % pTotal);
                    int ypoint = xpoint / pWidth;
                    xpoint = xpoint % pWidth;

                    switch (x / pTotal)
                    {
                        case 0:
                            return (double)flagBitmaps[i].GetPixel(xpoint, ypoint).R / 255;
                        case 1:
                            return (double)flagBitmaps[i].GetPixel(xpoint, ypoint).G / 255;
                        default:
                            return (double)flagBitmaps[i].GetPixel(xpoint, ypoint).B / 255;
                    }
                });
            }
            
            //PiwotBrainLib.BrainCore bc = new PiwotBrainLib.BrainCore(pTotal * 3, new int[5] { 128, 64, 32, 64, 128 }, pTotal * 3);

            PiwotBrainLib.BrainCore bc = new PiwotBrainLib.BrainCore("brain_mid_103.txt");
            bc.StreachLayer(0, 2, 2, 12, 0.25);

            bc.StreachLayer(6, 2, 2, 12, 0.25);

            PiwotDrawingLib.UI.Containers.SimpleFunctionDisplay f = new PiwotDrawingLib.UI.Containers.SimpleFunctionDisplay(
                new Int2(Arit.Clamp(pb2.Position.X + pb2.Size.X + 1, 20, int.MaxValue), pb2.Position.Y), displaySize, "Moment Error", PiwotDrawingLib.Misc.Boxes.BoxType.doubled, ErrorFunc);

            PiwotDrawingLib.UI.Containers.SimpleFunctionDisplay f2 = new PiwotDrawingLib.UI.Containers.SimpleFunctionDisplay(
                f.Position + new Int2(0, f.Size.Y), displaySize, "Moment error zoom", PiwotDrawingLib.Misc.Boxes.BoxType.doubled, ErrorFunc2);

            PiwotDrawingLib.UI.Containers.SimpleFunctionDisplay f3 = new PiwotDrawingLib.UI.Containers.SimpleFunctionDisplay(
                f2.Position + new Int2(0, f2.Size.Y), displaySize, "History error", PiwotDrawingLib.Misc.Boxes.BoxType.doubled, ErrorFuncLong);

            PiwotDrawingLib.Misc.Boxes.DrawBox(PiwotDrawingLib.Misc.Boxes.BoxType.normal, 1, pHeight / 2 + 2, f.Position.X - 1, Renderer.WindowSize.Y - (pHeight / 2 + 2));

            PiwotBrainLib.LearningBrain learningBrain = new PiwotBrainLib.LearningBrain(bc)
            {
                ExampleBlockSize = 1,
                DataExtractor = RandomData,
                Accuracy = 100000,
                ErrorMemoryLenght = 200,
                Momentum = 0.1,
                NeuronActivation = new PiwotBrainLib.SechActivation()
            };
            
            Vector<double> ranfFlag;
            Stopwatch stopwatch = new Stopwatch();
            uint lTime = 0, cTime = 0;

            int saved = 0;
            long times = 0;
            
            while (true)
            {
                stopwatch.Restart();
                learningBrain.LearnBlocks(100);
                stopwatch.Stop();
                lTime = (uint)stopwatch.ElapsedMilliseconds;
                
                ranfFlag = RandomFlag();
                pb.Image = VectorToBitmap(ranfFlag);
                stopwatch.Restart();
                pb2.Image = VectorToBitmap(learningBrain.Calculate(ranfFlag));
                stopwatch.Stop();
                cTime = (uint)stopwatch.ElapsedMilliseconds;
                pb.RefreshContent();
                pb2.RefreshContent();
                
                memory[memoryPoint] = (float)learningBrain.MeanSquaredError;
                
                if(memoryLong.Contains(0))
                {
                    System.Console.ReadKey(true);
                }
                memoryPoint++;
                if (memoryPoint >= displaySize.X - 2)
                    memoryPoint = 0;
                GetMaxErr();
                Renderer.Draw($"AvgErr: {learningBrain.MeanSquaredError.ToString("0.0000")} -", 2, pHeight / 2 + 3);
                Renderer.Draw($"MaxErr: {maxErr.ToString("0.0000")} -", 2, pHeight / 2 + 4);
                Renderer.Draw($"MinErr: {minErr.ToString("0.0000")} -", 2, pHeight / 2 + 5);
                Renderer.Draw($"Max-Min: {(maxErr - minErr).ToString("0.0000")} -", 2, pHeight / 2 + 6);
                Renderer.Draw($"HMaxErr: {maxErrLong.ToString("0.0000")} -", 2, pHeight / 2 + 7);
                Renderer.Draw($"Learned: {learningBrain.ExamplesDone} -", 2, pHeight / 2 + 8);
                Renderer.Draw($"Learn T: {lTime} -", 2, pHeight / 2 + 9);
                Renderer.Draw($"Calc T: {cTime} -", 2, pHeight / 2 + 10);
                Renderer.Draw($"Acc: {learningBrain.Accuracy} -", 2, pHeight / 2 + 11);
                Renderer.Draw($"Mom: {learningBrain.Momentum} -", 2, pHeight / 2 + 12);
                f.Draw();
                f2.Draw();
                f3.Draw();
                f.RefreshContent();
                f2.RefreshContent();
                f3.RefreshContent();
                Renderer.ForcePrint();
                times++;
                if(times % 10 == 0)
                {
                    memoryLong.Add((float)learningBrain.MeanSquaredError);
                }
                if (times > 100)
                {
                    times = 0;
                    learningBrain.SaveToFile("", $"brain_mid_{saved}");
                    saved++;
                }

                //Console.ReadKey(true);
            }



        }
        
        static void GetMaxErr()
        {
            float x = 0;
            for (int i = 0; i < memory.Length; i++)
            {
                if (x < memory[i])
                    x = memory[i];
            }
            maxErr = x;
            x = float.MaxValue;
            for (int i = 0; i < memory.Length; i++)
            {
                if (x > memory[i])
                    x = memory[i];
            }
            minErr = x;

            x = 0;
            for (int i = 0; i < memoryLong.Count; i++)
            {
                if (x < memoryLong[i])
                    x = memoryLong[i];
            }
            maxErrLong = x;
        }
        static float ErrorFunc(float x)
        {
            
            int id = (int)System.Math.Round(x * memory.Length);

            id += memoryPoint;
            return ((memory[id % memory.Length])) / (maxErr);

        }

        static float ErrorFunc2(float x)
        {

            int id = (int)System.Math.Round(x * memory.Length);

            id += memoryPoint;

            return ((memory[id % memory.Length]) - minErr) / (maxErr - minErr);

        }

        static float ErrorFuncLong(float x)
        {
            if (memoryLong.Count == 0)
                return 0;
            int id = (int)System.Math.Floor(x * memoryLong.Count);


            return memoryLong[id] / maxErrLong;

        }

        static Bitmap VectorToBitmap(Vector<double> v)
        {
            Bitmap b = new Bitmap(pWidth, pHeight);
            int xpos = 0;
            int ypos = 0;
            for(int i = 0; i < pTotal; i++)
            {
                xpos = i % pWidth;
                ypos = i / pWidth;
                b.SetPixel(xpos, ypos, Color.FromArgb(255, Arit.Clamp((int)(v[i] * 255), 0, 255), Arit.Clamp((int)(v[i + pTotal] * 255), 0, 255), Arit.Clamp((int)(v[i + pTotal * 2] * 255), 0, 255)));
            }
            return b;
        }

        public static Vector<double> RandomFlag()
        {
            //return Vector<double>.Build.Dense(pTotal * 3, (r) => ((r * 2) / (pWidth * 3) == 0 ? 1 : 0));
            return flagMatrixes[Rand.Int(flagMatrixes.Length)];
        }
        public static (Matrix<double>, Matrix<double>) RandomData(PiwotBrainLib.LearningBrain lb)
        {
            Matrix<double> rng = RandomFlag().ToColumnMatrix();
            return (rng.Clone(), rng.Clone());
        }
    }
}
