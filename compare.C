//
// compare.C       20230615   Aiqiang Zhang 
//
// root -q -l -b 'compare.C("run1.root", "run2.root", "run1_run2.csv")'
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
using namespace std;
typedef struct {
    TH1* bsenergy;
    TH1* ovaq;
    TH2* r2z;
} Res;
Res readRes(string filename){
    cout<<filename<<endl;
    TFile *file = new TFile(filename.c_str(), "READ");
    //-----------------------------------------------------------------------------------
    TTree* tree = (TTree*)file->Get("data");

    TH1F *bsenergy = new TH1F("h12","bsenergy LINAC",80,0.,40.);
    TH1F *ovaq = new TH1F("h15","ovaq LINAC",100,0.,1.);
    TH2F *r2z = new TH2F("h22","r2 vs Z LINAC", 80,0,222.01,80,-16.1,+16.1);
    Res res = {bsenergy, bsenergy, r2z};

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


	if ((HEAD->idtgsk & 2**24) != 0) {
	    // LINAC
	    bsenergy->Fill(LOWE->bsenergy);
	    ovaq->Fill(bsovaq);
	    r2z->Fill(r2,z);
	}
   
    }
    cout << "done." << endl;
   
    return res;
}
void compare(string run1_filename, string run2_filename, string fout)
{
    //-------------------------------------------------------------
    // read lowe events
    //-------------------------------------------------------------
    Res res1 = readRes(run1_filename);
    Res res2 = readRes(run2_filename);

    ofstream out_f(fout.c_str(), ios::out);
    out_f << res1.bsenergy->KolmogorovTest(res2.bsenergy, "") << "," <<
      res1.ovaq->KolmogorovTest(res2.ovaq, "") << "," <<
      res1.r2z->KolmogorovTest(res2.r2z, "") << std::endl;
    out_f.close();
}

