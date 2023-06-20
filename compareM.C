//
// compareM.C       20230620   Aiqiang Zhang (Modified from Takeuchi)
//
// root -q -l -b 'compareM.C("run1.root", "run2.root", "run1_run2.csv")'
//
// The following setup files will be needed.
//
// .rootrc
// rootlogon.C
//

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <time.h>
#include "Util.h"
using namespace std;
vector<string> stringSplit(const TString& strIn, TString delim) {
    TObjArray* tx = strIn.Tokenize(delim);
    vector<string> elems;
    for(int i=0; i<tx->GetEntries(); i++){
        string x = ((TObjString *)(tx->At(i)))->String();
	  elems.push_back(x);
    }
    return elems;
};
void compareM(string run1_no, string run2_no, string run1_x, string run2_x, string phase1, string phase2, int Z, int E, string fout)
{
    // parse the args
    vector<string> run1_nos = stringSplit(run1_no, " ");
    vector<string> run2_nos = stringSplit(run2_no, " ");
    int run1_num = run1_nos.size();
    int run2_num = run2_nos.size();
    vector<string> run1_xs = stringSplit(run1_x, " ");
    vector<string> run2_xs = stringSplit(run2_x, " ");
    int run2_x_min=20, run2_x_max=-20;
    for(int i=0;i<run2_num; i++) {
	    int temp = TString(run2_xs[i]).Atoi();
	    if (run2_x_min>temp) run2_x_min = temp;
	    if (run2_x_max<temp) run2_x_max = temp;
    }
    Res test;
    Res* run1_fs = new Res[run1_num];
    Res* run2_fs = new Res[run2_num];
    int xsigma = 500, ysigma=500, zsigma=700;
    //-------------------------------------------------------------
    // read lowe events
    //-------------------------------------------------------------
    for(int i=0; i<run1_num; i++){
        run1_fs[i] = readRes(phase1 + run1_nos[i] + ".root");
    }
    for(int i=0; i<run2_num; i++){
        run2_fs[i] = readRes(phase2 + run2_nos[i] + ".root");
        //cout<<run2_fs[i].Vertexes[0]<<endl;
    }
    // plot
    TH1*** thArray= new TH1**[run2_num+run1_num];
    vector<TString> labels(run2_num+run1_num);
    TString XTitles[12] = {"X[cm]", "Y[cm]", "Z[cm]", "X", "Y", "Z", "Neff", "bsGoodness", "bsDirectionKS", "bsEnergy", "ovaq", "R^{2}[m^{2}]"};
    TString YTitles[12] = {"Entries", "Entries", "Entries", "Entries", "Entries", "Entries", "Entries", "Entries", "Entries", "Entries", "Entries", "Z[m]"};
    int upperV[12] = {run2_x_max*100+xsigma, ysigma, Z*100+zsigma, 1, 1, 1, 0, 0, 0, 0, 0, 0};
    int lowerV[12] = {run2_x_min*100-xsigma, -ysigma, Z*100-zsigma, -1, -1, -1, 0, 0, 0, 0, 0, 0};
    for(int i=0; i<run2_num; i++){
	    thArray[i] = new TH1*[12];
	    thArray[i][0] = run2_fs[i].Vertexes[0];
	    thArray[i][1] = run2_fs[i].Vertexes[1];
            thArray[i][2] = run2_fs[i].Vertexes[2];
	    thArray[i][3] = run2_fs[i].Directions[0];
	    thArray[i][4] = run2_fs[i].Directions[1];
            thArray[i][5] = run2_fs[i].Directions[2];
            thArray[i][6] = run2_fs[i].Neff;
            thArray[i][7] = run2_fs[i].bsGoodness;
            thArray[i][8] = run2_fs[i].bsDirectionKS;
	    thArray[i][9] = run2_fs[i].bsenergy;
            thArray[i][10] = run2_fs[i].ovaq;
            thArray[i][11] = run2_fs[i].r2z;

	    labels[i] = TString(run2_nos[i]+"_X="+run2_xs[i]+"m");
	    for(int j=0; j<11; j++){
	        thArray[i][j]->SetLineColor(i+run1_num+1);
	    }
    }
    for(int i=0; i<run1_num; i++){
	    thArray[run2_num+i] = new TH1*[12];
	    thArray[run2_num+i][0] = run1_fs[i].Vertexes[0];
	    thArray[run2_num+i][1] = run1_fs[i].Vertexes[1];
            thArray[run2_num+i][2] = run1_fs[i].Vertexes[2];
	    thArray[run2_num+i][3] = run1_fs[i].Directions[0];
	    thArray[run2_num+i][4] = run1_fs[i].Directions[1];
            thArray[run2_num+i][5] = run1_fs[i].Directions[2];
            thArray[run2_num+i][6] = run1_fs[i].Neff;
            thArray[run2_num+i][7] = run1_fs[i].bsGoodness;
            thArray[run2_num+i][8] = run1_fs[i].bsDirectionKS;
	    thArray[run2_num+i][9] = run1_fs[i].bsenergy;
            thArray[run2_num+i][10] = run1_fs[i].ovaq;
            thArray[run2_num+i][11] = run1_fs[i].r2z;

            labels[run2_num+i] = TString(run1_nos[i]+"_X="+run1_xs[i]+"m");
    }

    TLegend* legend;
    TCanvas *c1 = new TCanvas("c1", "c1", 900, 1200);
    c1->Divide(3, 4);
    for(int j=0; j<11; j++){
        c1->cd(j+1);
        legend = new TLegend(0.6, 0.7, 0.95, 0.95);
        for(int i=0; i<(run2_num+run1_num); i++){
	    thArray[i][j]->Draw("same");
            legend->AddEntry(thArray[i][j], labels[i], "l");
	}
        thArray[0][j]->SetStats(0);
        thArray[0][j]->GetXaxis()->SetTitle(XTitles[j]);
	if(lowerV[j]!=upperV[j])
            thArray[0][j]->GetXaxis()->SetRangeUser(lowerV[j], upperV[j]);
        c1->GetPad(j+1)->SetLogy(1);
        legend->Draw();
    }
    c1->Update();
    c1->Print((fout+"(").c_str(), "pdf");
    
    TCanvas *c2 = new TCanvas("c2", "c2", 900, 1200);
    c2->Divide(3, 4);
    for(int j=0; j<11; j++){
        c2->cd(j+1);
        legend = new TLegend(0.6, 0.7, 0.95, 0.95);
        for(int i=0; i<(run2_num+run1_num); i++){
	    thArray[i][j]->Draw("same");
            legend->AddEntry(thArray[i][j], labels[i], "l");
	}
        thArray[0][j]->SetStats(0);
        thArray[0][j]->GetXaxis()->SetTitle(XTitles[j]);
	if(lowerV[j]!=upperV[j])
            thArray[0][j]->GetXaxis()->SetRangeUser(lowerV[j], upperV[j]);
        legend->Draw();
    }
    c2->Update();
    c2->Print(fout.c_str(), "pdf");

    TCanvas *c2d = new TCanvas("c2d", "c2d", 1200, 900);
    c2d->Divide(4,3);
    for(int j=11; j<12; j++)
        for(int i=0; i<(run2_num+run1_num); i++){
            c2d->cd(i+1);
            thArray[i][j]->SetOption("colz");
	    thArray[i][j]->SetXTitle(XTitles[j]);
	    thArray[i][j]->SetYTitle(YTitles[j]);
	    thArray[i][j]->SetTitle(labels[i]);
	    thArray[i][j]->Draw();
	}
    c2d->Update();
    c2d->Print((fout+")").c_str(), "pdf");
    /*ofstream out_f(fout.c_str(), ios::out);
    out_f << res1.bsenergy->KolmogorovTest(res2.bsenergy, "") << "," <<
      res1.ovaq->KolmogorovTest(res2.ovaq, "") << "," <<
      res1.r2z->KolmogorovTest(res2.r2z, "") << std::endl;
    out_f.close();
    */
}

