#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <TLatex.h>
#include <TAxis.h>
#include <TLegend.h>
#include <TCanvas.h>
#include <TLine.h>
#include <TH1F.h>
#include <TGraphErrors.h>
#include <TMultiGraph.h>
#include <TStyle.h>
#include <TMath.h>

#define N_DATA 2

Int_t main(const int argc,char *argv[]){

   // Check arguments
   if (argc != 5) {
      printf("Usage: ./compare_all [run_first] [run_last] [dir name] [linac dat sum]\n");
      return 0;
   }
   Int_t run_first = atoi(argv[1]);
   Int_t run_last  = atoi(argv[2]);
   TString dir = argv[3];
   TString sumdat = argv[4];
   // Canvas Style
   gStyle->SetFrameBorderMode(0);
   gStyle->SetCanvasBorderMode(0);
   gStyle->SetPadColor(kWhite);
   gStyle->SetCanvasColor(kWhite);
   gStyle->SetFillColor(kWhite);
   gStyle->SetOptStat(0);
   gStyle->SetPadGridX(1);
   gStyle->SetPadGridY(1);

   //Font Style
   gStyle->SetTextFont(132);
   gStyle->SetLabelFont(132,"XYZ");
   gStyle->SetTitleFont(132,"XYZ");

   //Title and Label Style
   gStyle->SetTitleSize(0.09,"XY");
   gStyle->SetLabelSize(0.07,"XY");
   gStyle->SetNdivisions(505,"X");
   gStyle->SetNdivisions(510,"Y");
   gStyle->SetLegendBorderSize(0);
   gStyle->SetMarkerStyle(20);
   gStyle->SetTitleXOffset(0.8);
   gStyle->SetTitleYOffset(0.4);
   gStyle->SetPadTopMargin(0.06);
   gStyle->SetPadLeftMargin(0.08);
   gStyle->SetPadRightMargin(0.02);
   gStyle->SetPadBottomMargin(0.19);

   // Make graphs
   TGraphErrors* gr_neff[N_DATA]; // Neff vs. run number
   TGraphErrors* gr_diff[N_DATA]; // Neff Data-MC diff vs. run number. First graph in the array will not be used.
   TGraphErrors* gr_neff_sel[N_DATA]; // Neff vs. run number for selected runs
   TGraphErrors* gr_ratio_sel[N_DATA]; // Neff MC/Data ratio vs. run number for selected runs
   TMultiGraph* mgr_neff = new TMultiGraph();
   TMultiGraph* mgr_diff = new TMultiGraph();
   TMultiGraph* mgr_neff_sel = new TMultiGraph();
   TMultiGraph* mgr_ratio_sel = new TMultiGraph();
   for (Int_t iData=0; iData<N_DATA; iData++) {
      gr_neff[iData] = new TGraphErrors();
      gr_diff[iData] = new TGraphErrors();
      gr_neff_sel[iData] = new TGraphErrors();
      gr_ratio_sel[iData] = new TGraphErrors();
      mgr_neff->Add(gr_neff[iData]);
      mgr_neff_sel->Add(gr_neff_sel[iData]);
      if (iData > 0) {
         mgr_diff->Add(gr_diff[iData]);
         mgr_ratio_sel->Add(gr_ratio_sel[iData]);
      }
      gr_neff[iData]->SetMarkerColor(iData+1);
      gr_diff[iData]->SetMarkerColor(iData+1);
      gr_neff_sel[iData]->SetMarkerColor(iData+1);
      gr_ratio_sel[iData]->SetMarkerColor(iData+1);
      gr_neff[iData]->SetLineColor(iData+1);
      gr_diff[iData]->SetLineColor(iData+1);
      gr_neff_sel[iData]->SetLineColor(iData+1);
      gr_ratio_sel[iData]->SetLineColor(iData+1);
   }

   // Open run sumamry file
   std::ifstream ifs_runsum(sumdat);
   // std::ifstream ifs_runsum("/home/sklowe/linac/const/linac_runsum.dat");

   // Loop for runs in run summary file
   std::string line;
   std::vector<Int_t> list_sel;
   while (getline(ifs_runsum, line, '\n')) {

      // Read line
      std::stringstream ss(line);
      Int_t run_this, e_mode, run_mode;
      ss >> run_this >> e_mode >> run_mode;

      // Select runs
      if(run_mode != 0 || run_this < run_first || run_this > run_last){
         continue;
      }

      // Open text file for this run
      TString filename = Form("%s/linac_run%06d.dat", dir.Data(), run_this);
      std::ifstream ifs_data(filename.Data());
      if(!ifs_data){
         std::cout << filename << " does not exist." << std::endl;
         return 1;
      }

      // Read text file
      Float_t neff[N_DATA];
      Float_t neff_err[N_DATA];
      Float_t sigma;
      for(Int_t j=0; j<5; j++){
      for (Int_t iData=0; iData<N_DATA; iData++) {
         ifs_data >> neff[iData] >> neff_err[iData] >> sigma;
      }}

      // Loop for data types (Data, Detsim, SKG4 ...)
      for (Int_t iData=0; iData<N_DATA; iData++) {

         // Set points to graphs
         gr_neff[iData]->SetPoint(gr_neff[iData]->GetN(), run_this+iData, neff[iData]);
         gr_neff[iData]->SetPointError(gr_neff[iData]->GetN()-1, 0., neff_err[iData]);
         if (iData > 0 && neff[0] > 0.) {  // For MC
            Float_t diff = (neff[iData] - neff[0]) / neff[0] * 100.0;
            Float_t diff_err = diff * sqrt( pow(neff_err[iData]/neff[iData], 2) + pow(neff_err[0]/neff[0], 2) );
            gr_diff[iData]->SetPoint(gr_diff[iData]->GetN(), run_this+iData, diff);
            gr_diff[iData]->SetPointError(gr_diff[iData]->GetN()-1, 0., diff_err);
         }

         // Set points to graphs for selected runs
         // if (e_mode == 8) {
         if (true) {
            if (iData==0) {
               list_sel.push_back(run_this);
            }
            gr_neff_sel[iData]->SetPoint(gr_neff_sel[iData]->GetN(), gr_neff_sel[iData]->GetN()+0.2*iData, neff[iData]);
            gr_neff_sel[iData]->SetPointError(gr_neff_sel[iData]->GetN()-1, 0., neff_err[iData]);
            if (iData > 0 && neff[0] > 0.) {  // For MC
               Float_t ratio = neff[iData]/neff[0];
               Float_t ratio_err = ratio * sqrt( pow(neff_err[iData]/neff[iData], 2) + pow(neff_err[0]/neff[0], 2) );
               gr_ratio_sel[iData]->SetPoint(gr_ratio_sel[iData]->GetN(), gr_ratio_sel[iData]->GetN()+0.2*(iData-1), ratio);
               gr_ratio_sel[iData]->SetPointError(gr_ratio_sel[iData]->GetN()-1, 0., ratio_err);
            }
         } 

      } // End data loop

   } // End run loop
   ifs_runsum.close();

   // Make canvases
   TCanvas *c1 = new TCanvas("c1","c1",700,500);
   TCanvas *c2 = new TCanvas("c2","c2",700,500);
   c1->Divide(1,2);
   c2->Divide(1,2);

   // Draw Neff vs runnum
   c1->cd(1);
   mgr_neff->SetTitle(";Run number;Neff");
   mgr_neff->Draw("APE");

   // Add legend
   TLegend *leg1 = new TLegend(0.1,0.6,0.2,0.8);
   leg1->SetFillStyle(0);
   TString str_data[N_DATA] = {"DATA", "MC"};
   for (Int_t iData=0; iData<N_DATA; iData++) {
      leg1->AddEntry(gr_neff[iData], str_data[iData], "lp");
   }
   leg1->Draw();

   // Draw Neff diff vs. runnum
   c1->cd(2);
   mgr_diff->SetTitle(";Run number;(MC-Data)/MC*100");
   mgr_diff->Draw("APE");
   mgr_diff->GetYaxis()->SetRangeUser(-6.0,6.0);

   // Draw Neff for selected runs
   c2->cd(1);
   mgr_neff_sel->Draw("APE");
   mgr_neff_sel->SetTitle(";Run number;Neff");
   mgr_neff_sel->GetXaxis()->SetTitleOffset(1.1);
   mgr_neff_sel->GetXaxis()->LabelsOption("v");

   // Add legend
   TLegend *leg2 = new TLegend(0.65,0.7,0.95,0.85);
   leg2->SetFillStyle(0);
   for (Int_t iData=0; iData<N_DATA; iData++) {
      leg2->AddEntry(gr_neff_sel[iData], str_data[iData], "lp");
   }
   leg2->Draw();

   // Draw Neff ratio for selected runs
   c2->cd(2); 
   mgr_ratio_sel->SetTitle(";Run number;MC/Data");
   mgr_ratio_sel->Draw("APE");

   // Set X axis labels
   for (UInt_t iRun=0; iRun<list_sel.size(); iRun++) {
      Int_t bin_neff = mgr_neff_sel->GetXaxis()->FindBin(iRun);
      mgr_neff_sel->GetXaxis()->SetBinLabel(bin_neff, Form("%d", list_sel[iRun]));
      Int_t bin_ratio = mgr_ratio_sel->GetXaxis()->FindBin(iRun);
      mgr_ratio_sel->GetXaxis()->SetBinLabel(bin_ratio, Form("%d", list_sel[iRun]));
   }
   mgr_ratio_sel->GetXaxis()->SetTitleOffset(1.1);
   mgr_ratio_sel->GetXaxis()->LabelsOption("v");
   // mgr_neff_sel->GetXaxis()->LabelsOption("h");
   // mgr_ratio_sel->GetXaxis()->LabelsOption("h");

   // Draw lines in the ratio graph
   TLine* lin = new TLine();
   lin->SetLineColor(1);
   lin->SetLineWidth(3);
   Float_t xmin_graph = mgr_ratio_sel->GetXaxis()->GetXmin();
   Float_t xmax_graph = mgr_ratio_sel->GetXaxis()->GetXmax();
   lin->DrawLine(xmin_graph, 1.00, xmax_graph, 1.00);
   lin->SetLineStyle(2);
   lin->SetLineWidth(2);
   lin->DrawLine(xmin_graph, 1.01, xmax_graph, 1.01);
   lin->DrawLine(xmin_graph, 0.99, xmax_graph, 0.99);

   // Open output file
   std::ofstream ofs(Form("%s/corepmt.dat", dir.Data()), std::ios::app);

   // Legend to draw mean and RMS
   TLatex *latex = new TLatex();
   latex->SetTextSize(0.04);

   // Calculate and draw mean and RMS
   for (Int_t iData=1; iData<N_DATA; iData++) {

      // Calculate mean and RMS
      Float_t ratio_mean = TMath::Mean(gr_ratio_sel[iData]->GetN(), gr_ratio_sel[iData]->GetY());
      Float_t ratio_rms = TMath::RMS(gr_ratio_sel[iData]->GetN(), gr_ratio_sel[iData]->GetY());

      // Draw
      latex->SetTextColor(iData+1);
      latex->DrawTextNDC(0.75, 0.80-0.05*iData, Form("%s Mean = %5.4lf  RMS = %5.4lf", str_data[iData].Data(), ratio_mean, ratio_rms));

      // Write
      printf("%s/Data Mean \t\t:=  %5.4lf  RMS = %5.4lf\n", str_data[iData].Data(), ratio_mean, ratio_rms);
      ofs << ratio_mean << " " << ratio_rms << " ";
   }

   ofs << std::endl;
   ofs.close();

   // Save as PDF
   c1->Update();
   c1->Print(Form("%s/compare_all.pdf(", dir.Data()));
   c2->Update();
   c2->Print(Form("%s/compare_all.pdf)", dir.Data()));
   c2->SaveAs("test.root");

   return 0; 

}
