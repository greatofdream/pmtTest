#include <fstream>
#include <sstream>
#include <cmath>
#include <TFile.h>
#include <TH1F.h>
#include <TLatex.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TF1.h>
#include <TLegend.h>
#include "SuperManager.h"

#define N_DATA 2
#define N_HIST 12

int main(int argc,char *argv[]){

   if (argc!=6) {
      std::cout << "Usage: ./bin/compare [dir data] [dir MC] [dir out] [Run number] [linac sum dat]" << std::endl;
      return 1;
   }
   TString dir_data = argv[1];
   TString dir_mc = argv[2];
   TString dir_out = argv[3];
   Int_t runnum = atoi(argv[4]);
   TString sumdat = argv[5];
   //************* Set styles for drawing **************//

   // Canvas Style  
   gStyle->SetOptStat(0);
   gStyle->SetPadBottomMargin(0.15);

   // Font Style
   gStyle->SetTextFont(132);
   gStyle->SetLabelFont(132,"XYZ");
   gStyle->SetTitleFont(132,"XYZ");

   // Title and Label Style
   gStyle->SetTitleSize(0.08,"X");
   gStyle->SetTitleSize(0.08,"Y");
   gStyle->SetLabelSize(0.075,"Y");
   gStyle->SetLabelSize(0.075,"X");
   gStyle->SetTitleXOffset(0.9);
   gStyle->SetTitleYOffset(1.5);
   gStyle->SetNdivisions(505,"X");
   gStyle->SetNdivisions(510,"Y");
   gStyle->SetStripDecimals(false);

   //************* Get run summary information **************//

   // Open runsum file  
   std::ifstream ifs(sumdat);
   //std::ifstream ifs("/home/sklowe/linac/const/linac_runsum.dat");   
   // Read runsum file
   Int_t linac_run, e_mode, dummy;
   Float_t beam_pos[3];
   std::string line;
   while (getline(ifs, line, '\n')) {
      // Read line
      std::stringstream ss(line);
      ss >> linac_run >> e_mode >> dummy >> beam_pos[0] >> beam_pos[1] >> beam_pos[2] >> dummy;
      if(linac_run == runnum){
         std::cout<<linac_run<<std::endl;
         break;
      }
   }
   if(linac_run != runnum) {
      std::cerr<< "Run" << runnum << " was not found in runsum.dat."<<std::endl;
      return 1;
   }
   ifs.close();

   //************* Fill histograms **************//

   // Make histograms
   TH1F *hist[N_DATA][N_HIST];
   for (Int_t iData=0; iData<N_DATA; iData++) {
      hist[iData][0] = new TH1F(Form("h%d_vertex_x", iData), ";Vertex x [cm]", 1000, -2500.0, +2500.0);
      hist[iData][1] = new TH1F(Form("h%d_vertex_y", iData), ";Vertex y [cm]", 1000, -2500.0, +2500.0);
      hist[iData][2] = new TH1F(Form("h%d_vertex_z", iData), ";Vertex z [cm]", 1000, -2500.0, +2500.0);
      hist[iData][3] = new TH1F(Form("h%d_bsenergy", iData), ";Energy [MeV]", 160, 0.0, 40.0);
      hist[iData][4] = new TH1F(Form("h%d_neff", iData), ";Neff hit", 150, 0.0, 300.0);
      hist[iData][5] = new TH1F(Form("h%d_angle", iData), ";Angle [deg]", 90, 0.0, 180.0);
      hist[iData][6] = new TH1F(Form("h%d_ovaq", iData), ";Ovaq", 50, 0.0, 1.0);
      hist[iData][7] = new TH1F(Form("h%d_bsgood", iData), ";BS goodness", 100, 0.4, 1.0);
      hist[iData][8] = new TH1F(Form("h%d_patlik", iData), ";Patlik", 50, -2.5, 1.0);
      hist[iData][9] = new TH1F(Form("h%d_dir_x", iData), ";Dir X", 100, -1.0, 1.0);
      hist[iData][10] = new TH1F(Form("h%d_dir_y", iData), ";Dir Y", 100, -1.0, 1.0);
      hist[iData][11] = new TH1F(Form("h%d_dir_z", iData), ";Dir Z", 100, -1.0, 1.0);
   }

   // List of directories
   TString dir_list[N_DATA] = {dir_data, dir_mc};
 
   // Loop for data
   for(Int_t iData=0; iData<N_DATA; iData++){

      // Make tree manager
      Int_t id = 10;
      SuperManager* Smgr = SuperManager::GetManager(); 
      Smgr->CreateTreeManager(id,"\0","\0",2);  
      TreeManager* mgr = Smgr->GetTreeManager(id);
      
      // Set input file
      TChain c_dummy("data");
      c_dummy.Add(Form("%s/%06d/*.root", dir_list[iData].Data(), runnum));
      TObjArray* file_list = c_dummy.GetListOfFiles();
      for (Int_t iData=0; iData<file_list->GetEntries(); iData++) {
         mgr->SetInputFile(file_list->At(iData)->GetTitle());
      }

      // Initialize
      mgr->Initialize();
  
      // Set branch status
      TTree* tree = mgr->GetTree();
      tree->SetBranchStatus("*", 0);
      tree->SetBranchStatus("LOWE");
  
      // Get LOWE information
      LoweInfo *LOWE = mgr->GetLOWE();
  
      // Loop for entries
      Int_t nEntries = tree->GetEntries();  
      for(Int_t iEntry=0; iEntry<nEntries; iEntry++){

         // Get entry
         tree->GetEntry(iEntry);
    
         // Calculate parameters
         Float_t angle = acos(-1.0*LOWE->bsdir[2]);
         Float_t ovaq = LOWE->bsgood[1]*LOWE->bsgood[1]-LOWE->bsdirks*LOWE->bsdirks;

         // Fill histograms    
         hist[iData][0]->Fill(LOWE->bsvertex[0]);
         hist[iData][1]->Fill(LOWE->bsvertex[1]);
         hist[iData][2]->Fill(LOWE->bsvertex[2]);
         hist[iData][3]->Fill(LOWE->bsenergy);  
         hist[iData][4]->Fill(LOWE->bseffhit[0]);
         hist[iData][5]->Fill(angle*180.0/3.14); 
         hist[iData][6]->Fill(ovaq);
         hist[iData][7]->Fill(LOWE->bsgood[1]);
         hist[iData][8]->Fill(LOWE->bspatlik);
         hist[iData][9]->Fill(LOWE->bsdir[0]);
         hist[iData][10]->Fill(LOWE->bsdir[1]);
         hist[iData][11]->Fill(LOWE->bsdir[2]);
    
      } // End loop for entries
  
      // Close file
      Smgr->DeleteTreeManager(id);
  
   } // End loop for data

   SuperManager::DestroyManager();
   std::cout<<"begin Fit"<<std::endl;
  
   //************* Fit Neff histograms **************//

   // Fitting function
   TF1 *fgaus = new TF1("fgaus","gaus");

   // Output file
   std::ofstream ofs(Form("%s/linac_run%06d.dat", dir_out.Data(), runnum));

   // Loop for data
   for(Int_t j=0; j<5; j++){
   for(Int_t iData=0; iData<N_DATA; iData++){
   
      // Set parameters
      TH1F* h_this = hist[iData][j];
      Float_t cnst = h_this->GetMaximum();
      Int_t maxbin = h_this->GetMaximumBin();
      Float_t mean = h_this->GetXaxis()->GetBinCenter(maxbin);
      Float_t sigma = h_this->GetRMS();
      Double_t par[3] = {cnst, mean, sigma};
      fgaus->SetParameters(par);
   
      // Fit Neff histograms
      for (Int_t iFit=0; iFit<2; iFit++) {
         h_this->Fit(fgaus,"Q0", "", par[1] - par[2], par[1] + par[2]);
         fgaus->GetParameters(par);
      }
   
      // Output
      ofs << par[1] << " " << fgaus->GetParError(1) << " " << par[2] << " ";
   
   }
   ofs<<std::endl;
   }
   ofs.close();

   TFile* outfile = new TFile(Form("%s/linac_run%06d.root", dir_out.Data(), runnum), "RECREATE");
  for(Int_t iData=1; iData<N_DATA; iData++){
  for(Int_t iHist=0; iHist<N_HIST;iHist++) {
    hist[iData][iHist]->Write();
    }
  }
  outfile->Close();
   //************* Draw histograms **************//

   // Scale MC histograms to Data  
   Float_t nentries_data = hist[0][0]->GetEntries();
   for (Int_t iData=1; iData<N_DATA; iData++) {
      Float_t nentries_mc = hist[iData][0]->GetEntries();
      for (Int_t iHist=0; iHist<N_HIST; iHist++) {
         hist[iData][iHist]->Sumw2();
         hist[iData][iHist]->Scale(nentries_data/nentries_mc);
      }
   }

   // Legends
   TLegend *leg = new TLegend(0.7,0.5,0.9,0.7);
   leg->AddEntry(hist[0][0],"DATA","l");
   leg->AddEntry(hist[1][0],"MC","l");
   leg->SetBorderSize(0);
   leg->SetTextSize(0.06);
   leg->SetFillStyle(0);
   TLatex latex;
   latex.SetTextSize(0.062);
    
   // Draw
   TCanvas *c1 = new TCanvas("c1","c1",800,800);
   c1->Divide(3,4);
   for(Int_t iHist=0;iHist<N_HIST;iHist++){
      c1->cd(iHist+1);

      if( iHist < 5 ){
         Float_t x_min = hist[1][iHist]->GetMean() - 4.0*hist[1][iHist]->GetRMS();
         Float_t x_max = hist[1][iHist]->GetMean() + 4.0*hist[1][iHist]->GetRMS();
         hist[1][iHist]->GetXaxis()->SetRangeUser(x_min, x_max);
      }

      for (Int_t iData=0; iData<N_DATA; iData++) {
         hist[iData][iHist]->SetLineColor(iData+1);
      }

      hist[1][iHist]->Draw("hist");
      hist[0][iHist]->Draw("same e");

      // Draw legends
      latex.DrawTextNDC(0.68, 0.86, Form("%d MeV", e_mode));
      latex.DrawTextNDC(0.68, 0.78, Form("X = %4.2fm", beam_pos[0]/100.));
      latex.DrawTextNDC(0.68, 0.72, Form("Z = %4.2fm", beam_pos[2]/100.));
      leg->Draw();

   }
   hist[1][1]->SetTitle(Form("Run%06d", runnum));
   c1->Update();
   c1->Print(Form("%s/linac_run%06d.pdf", dir_out.Data(), runnum));

   return 0;
}
