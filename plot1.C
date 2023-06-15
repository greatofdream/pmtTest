//
// plot1.C       28-JUL-2021    y.takeuchi
//
// root -q -l -b 'plot1.C("/disk03/lowe10/sk7/tmp/lin/091368/lin.091368.000001.root")'
//
// The following setup files will be needed.
//
// ~sklowe/.rootrc
// ~sklowe/root/rootlogon.C
//

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <time.h>
using namespace std;

void plot1(char *filename, string fout)
{

    //-------------------------------------------------------------
    //make output file name
    //-------------------------------------------------------------
    // string fout = filename;
    string data_dir = "/disk03/lowe10/sk7/tmp/lin/";
    string data_ext = ".root";
    // string plot_dir = "/home/sklowe/public_html/linac23/plot/";
    // string plot_ext = ".png";
    
    // fout.replace(fout.find(data_dir), data_dir.length(), plot_dir);
    // fout.replace(fout.find(data_ext), data_ext.length(), plot_ext);
 
    cout << "Output file: " << fout << endl;

    //-------------------------------------------------------------
    //make histograms
    //-------------------------------------------------------------
    TH1F *h11 = new TH1F("h11","bsenergy w/o LINAC*",80,0.,40.);
    TH1F *h12 = new TH1F("h12","bsenergy LINAC",80,0.,40.);
    TH1F *h13 = new TH1F("h13","bsenergy LINAC-RF",80,0.,40.);

    TH1F *h14 = new TH1F("h14","ovaq w/o LINAC*",100,0.,1.);
    TH1F *h15 = new TH1F("h15","ovaq LINAC",100,0.,1.);
    TH1F *h16 = new TH1F("h16","ovaq LINAC-RF",100,0.,1.);

    TH2F *h21 = new TH2F("h21","r2 vs Z w/o LINAC*", 80,0,222.01,80,-16.1,+16.1);
    TH2F *h22 = new TH2F("h22","r2 vs Z LINAC", 80,0,222.01,80,-16.1,+16.1);
    TH2F *h23 = new TH2F("h23","r2 vs Z LINAC-RF", 80,0,222.01,80,-16.1,+16.1);

    //-------------------------------------------------------------
    // read lowe events
    //-------------------------------------------------------------

    //-----------------------------------------------------------------------------------
    //TFile *file = new TFile("/disk02/lowe8/sk6/lin/lin.081810.root","READ");
    //TFile *file = new TFile("/disk02/lowe8/sk6/lin/081810/lin.081810.000001.root","READ");
    TFile *file = new TFile(filename,"READ");
    //-----------------------------------------------------------------------------------
    TTree* tree = (TTree*)file->Get("data");
    
    LOWE = new LoweInfo;
    TBranch* lowebranch;
    HEAD = new Header;
    TBranch* headerbranch;
    TQREAL = new TQReal;
    TBranch* tqrealbranch;

    tree->SetBranchAddress("LOWE" , &LOWE , &lowebranch );
    tree->SetBranchAddress("HEADER", &HEAD , &headerbranch );
    tree->SetBranchAddress("TQREAL", &TQREAL , &tqrealbranch );

    int ntotal = tree->GetEntries();
    cout << "Filling: ntotal = " << ntotal << endl;

    float  bswall, bsovaq, bseffwal, bsclik, amsg, poswal[3], dist;
    double r2, z;
    int nev1=0, nev2=0, nev3=0, nev4=0, nev5=0, nev6=0, ntrigsk;

    Long64_t ii;

    for (int i = 0; i < ntotal; i++) {

        headerbranch->GetEntry(i);
	lowebranch->GetEntry(i);
	tqrealbranch->GetEntry(i);

	dist          = *reinterpret_cast<float*>(&LOWE->linfo[3]);
	bseffwal      = *reinterpret_cast<float*>(&LOWE->linfo[5]);
	bswall        = *reinterpret_cast<float*>(&LOWE->linfo[9]);
	bsclik        = *reinterpret_cast<float*>(&LOWE->linfo[24]);
	bsovaq        = *reinterpret_cast<float*>(&LOWE->linfo[25]);
	amsg          = *reinterpret_cast<float*>(&LOWE->linfo[29]);
	poswal[0]     = *reinterpret_cast<float*>(&LOWE->linfo[33]);
	poswal[1]     = *reinterpret_cast<float*>(&LOWE->linfo[34]);
	poswal[2]     = *reinterpret_cast<float*>(&LOWE->linfo[35]);

	r2 = ((LOWE->bsvertex[0]*LOWE->bsvertex[0])+(LOWE->bsvertex[1]*LOWE->bsvertex[1]))/10000.0;
	z = LOWE->bsvertex[2]/100.0;

	ntrigsk = (TQREAL->it0xsk - HEAD->t0) / 1.92 / 10;
//	cout <<  ntrigsk  << endl;
	    
//	if (bsovaq < 0.100)  continue;
//	if (LOWE->ltimediff/1000.00 < 20.00)  continue;
//	if (bswall < 150.00) continue;
//	if (LOWE->bsenergy < 5.00) continue;
//      if (LOWE->bsenergy > 10.00) continue;

//	cout << hex   << " 0x" << HEAD->idtgsk  << " " << (HEAD->idtgsk & 2**24) << " " << (HEAD->idtgsk & 2**25) << dec  << endl;

	if ((HEAD->idtgsk & 2**24) == 0 && (HEAD->idtgsk & 2**25) == 0) {
	    h11->Fill(LOWE->bsenergy);
	    h14->Fill(bsovaq);
	    h21->Fill(r2,z);
	}

	if ((HEAD->idtgsk & 2**24) != 0) {
	    // LINAC
	    h12->Fill(LOWE->bsenergy);
	    h15->Fill(bsovaq);
	    h22->Fill(r2,z);
	}
	if ((HEAD->idtgsk & 2**25) != 0) {
	    // LINAC microwave
	    h13->Fill(LOWE->bsenergy);
	    h16->Fill(bsovaq);
	    h23->Fill(r2,z);
	}
   
    }
    cout << "done." << endl;
   
  
    //-------------------------------------------------------------
    // draw
    //-------------------------------------------------------------
    gStyle->SetPadTopMargin(0.01);
    gStyle->SetPadLeftMargin(0.01);
    gStyle->SetPadRightMargin(0.01);
    gStyle->SetPadBottomMargin(0.01);
    gStyle->SetOptDate(1);
    gStyle->SetOptStat(10);
    gStyle->SetStatW(0.45);
    gStyle->SetStatH(0.30);

    TCanvas *c1 = new TCanvas("comp_bg","comp_bg",10,10,1200,1200);
    c1->Divide(3,3);
	
    double TSIZE = 0.04;
    double TOFF  = 1.1;
    double TOFF2  = 1.5;
	
    double TSIZE = 0.04;
    double TOFF  = 1.1;
    double TOFF2  = 1.5;

    c1->cd(1);
    gPad->SetTicks(1,1);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    gPad->SetRightMargin(0.15);
    gPad->SetTopMargin(0.10);
    c1->GetPad(1)->SetLogy(1);
    c1->GetPad(1)->SetGrid();

    h11->GetXaxis()->SetTitleSize(TSIZE);
    h11->GetXaxis()->SetLabelSize(TSIZE);
    h11->GetXaxis()->SetNdivisions(505,kTRUE);
    h11->GetYaxis()->SetTitleSize(TSIZE);
    h11->GetYaxis()->SetLabelSize(TSIZE);
    h11->SetXTitle("bsenergy");
    h11->SetLineColor(4);
    h11->DrawCopy("");

    c1->cd(2);
    gPad->SetTicks(1,1);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    gPad->SetRightMargin(0.15);
    gPad->SetTopMargin(0.10);
    c1->GetPad(2)->SetLogy(1);
    c1->GetPad(2)->SetGrid();

    h12->GetXaxis()->SetTitleSize(TSIZE);
    h12->GetXaxis()->SetLabelSize(TSIZE);
    h12->GetXaxis()->SetNdivisions(505,kTRUE);
    h12->GetYaxis()->SetTitleSize(TSIZE);
    h12->GetYaxis()->SetLabelSize(TSIZE);
    h12->SetXTitle("bsenergy");
    h12->SetLineColor(2);
    h12->DrawCopy("");

    c1->cd(3);
    gPad->SetTicks(1,1);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    gPad->SetRightMargin(0.15);
    gPad->SetTopMargin(0.10);
    c1->GetPad(3)->SetLogy(1);
    c1->GetPad(3)->SetGrid();

    h13->GetXaxis()->SetTitleSize(TSIZE);
    h13->GetXaxis()->SetLabelSize(TSIZE);
    h13->GetXaxis()->SetNdivisions(505,kTRUE);
    h13->GetYaxis()->SetTitleSize(TSIZE);
    h13->GetYaxis()->SetLabelSize(TSIZE);
    h13->SetXTitle("bsenergy");
    h13->SetLineColor(2);
    h13->DrawCopy("");

    c1->cd(4);
    gPad->SetTicks(1,1);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    gPad->SetRightMargin(0.15);
    gPad->SetTopMargin(0.10);
    c1->GetPad(4)->SetLogy(1);
    c1->GetPad(4)->SetGrid();

    h14->GetXaxis()->SetTitleSize(TSIZE);
    h14->GetXaxis()->SetLabelSize(TSIZE);
    h14->GetXaxis()->SetNdivisions(505,kTRUE);
    h14->GetYaxis()->SetTitleSize(TSIZE);
    h14->GetYaxis()->SetLabelSize(TSIZE);
    h14->SetXTitle("bsenergy");
    h14->SetLineColor(4);
    h14->DrawCopy("");

    c1->cd(5);
    gPad->SetTicks(1,1);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    gPad->SetRightMargin(0.15);
    gPad->SetTopMargin(0.10);
    c1->GetPad(5)->SetLogy(1);
    c1->GetPad(5)->SetGrid();

    h15->GetXaxis()->SetTitleSize(TSIZE);
    h15->GetXaxis()->SetLabelSize(TSIZE);
    h15->GetXaxis()->SetNdivisions(505,kTRUE);
    h15->GetYaxis()->SetTitleSize(TSIZE);
    h15->GetYaxis()->SetLabelSize(TSIZE);
    h15->SetXTitle("bsenergy");
    h15->SetLineColor(2);
    h15->DrawCopy("");

    c1->cd(6);
    gPad->SetTicks(1,1);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    gPad->SetRightMargin(0.15);
    gPad->SetTopMargin(0.10);
    c1->GetPad(6)->SetLogy(1);
    c1->GetPad(6)->SetGrid();

    h16->GetXaxis()->SetTitleSize(TSIZE);
    h16->GetXaxis()->SetLabelSize(TSIZE);
    h16->GetXaxis()->SetNdivisions(505,kTRUE);
    h16->GetYaxis()->SetTitleSize(TSIZE);
    h16->GetYaxis()->SetLabelSize(TSIZE);
    h16->SetXTitle("bsenergy");
    h16->SetLineColor(2);
    h16->DrawCopy("");

    c1->cd(7);
    gPad->SetTicks(1,1);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    gPad->SetRightMargin(0.15);
    gPad->SetTopMargin(0.10);
    c1->GetPad(7)->SetGrid();

    h21->SetOption("colz");
    h21->SetXTitle("R^{2}[m^{2}]");
    h21->GetXaxis()->SetTitleSize(TSIZE);
    h21->GetXaxis()->SetLabelSize(TSIZE);
    h21->GetXaxis()->SetTitleOffset(TOFF);
    h21->GetXaxis()->SetNdivisions(505,kTRUE);
    h21->SetYTitle("Z[m]");
    h21->GetYaxis()->SetTitleSize(TSIZE);
    h21->GetYaxis()->SetLabelSize(TSIZE);
    h21->GetYaxis()->SetTitleOffset(TOFF);
    h21->SetZTitle("Events");
    h21->GetZaxis()->SetTitleSize(TSIZE);
    h21->GetZaxis()->SetLabelSize(TSIZE);
    h21->GetZaxis()->SetTitleOffset(TOFF);
    h21->GetZaxis()->SetNdivisions(505,kTRUE);
    h21->DrawCopy("colz");

    c1->cd(8);
    gPad->SetTicks(1,1);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    gPad->SetRightMargin(0.15);
    gPad->SetTopMargin(0.10);
    c1->GetPad(8)->SetGrid();

    h22->SetOption("colz");
    h22->SetXTitle("R^{2}[m^{2}]");
    h22->GetXaxis()->SetTitleSize(TSIZE);
    h22->GetXaxis()->SetLabelSize(TSIZE);
    h22->GetXaxis()->SetTitleOffset(TOFF);
    h22->GetXaxis()->SetNdivisions(505,kTRUE);
    h22->SetYTitle("Z[m]");
    h22->GetYaxis()->SetTitleSize(TSIZE);
    h22->GetYaxis()->SetLabelSize(TSIZE);
    h22->GetYaxis()->SetTitleOffset(TOFF);
    h22->SetZTitle("Events");
    h22->GetZaxis()->SetTitleSize(TSIZE);
    h22->GetZaxis()->SetLabelSize(TSIZE);
    h22->GetZaxis()->SetTitleOffset(TOFF);
    h22->GetZaxis()->SetNdivisions(505,kTRUE);
    h22->DrawCopy("colz");

    c1->cd(9);
    gPad->SetTicks(1,1);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    gPad->SetRightMargin(0.15);
    gPad->SetTopMargin(0.10);
    c1->GetPad(9)->SetGrid();

    h23->SetOption("colz");
    h23->SetXTitle("R^{2}[m^{2}]");
    h23->GetXaxis()->SetTitleSize(TSIZE);
    h23->GetXaxis()->SetLabelSize(TSIZE);
    h23->GetXaxis()->SetTitleOffset(TOFF);
    h23->GetXaxis()->SetNdivisions(505,kTRUE);
    h23->SetYTitle("Z[m]");
    h23->GetYaxis()->SetTitleSize(TSIZE);
    h23->GetYaxis()->SetLabelSize(TSIZE);
    h23->GetYaxis()->SetTitleOffset(TOFF);
    h23->SetZTitle("Events");
    h23->GetZaxis()->SetTitleSize(TSIZE);
    h23->GetZaxis()->SetLabelSize(TSIZE);
    h23->GetZaxis()->SetTitleOffset(TOFF);
    h23->GetZaxis()->SetNdivisions(505,kTRUE);
    h23->DrawCopy("colz");

    c1->Print(fout.c_str());

}

