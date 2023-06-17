#include <iostream>
#include <fstream>
#include <string>
#include <TH1F.h>
using namespace std;
typedef struct {
    TH1F* bsenergy;
    TH1F* ovaq;
    TH2F* r2z;
    TH1F* Neff;
    TH1F* *Vertexes;
    TH1F* *Directions;
    TH1F* bsGoodness;
    TH1F* bsDirectionKS;
} Res;
Res readRes(string filename){
    cout<<filename<<endl;
    TFile *file = new TFile(filename.c_str(), "READ");
    //-----------------------------------------------------------------------------------
    TTree* tree = (TTree*)file->Get("data");

    TH1F *bsenergy = new TH1F("h12","bsenergy LINAC",80,0.,40.);
    TH1F *ovaq = new TH1F("h15","ovaq LINAC",100,0.,1.);
    TH2F *r2z = new TH2F("h22","r2 vs Z LINAC", 80,0,222.01,80,-16.1,+16.1);
    TH1F *Neff = new TH1F("Neff", "Neff LINAC", 150, 0, 300);
    TH1F **Vertexes = new TH1F*[3];
    Vertexes[0] = new TH1F("Vertex_X", "Vertex X LINAC", 1000, -2500, 2500);
    Vertexes[1] = new TH1F("Vertex_Y", "Vertex Y LINAC", 1000, -2500, 2500);
    Vertexes[2] = new TH1F("Vertex_Z", "Vertex Z LINAC", 1000, -2500, 2500);
    TH1F **Directions = new TH1F*[3];
    Directions[0] = new TH1F("Direction_X", "Direction X LINAC", 100, -1, 1);
    Directions[1] = new TH1F("Direction_Y", "Direction Y LINAC", 100, -1, 1);
    Directions[2] = new TH1F("Direction_Z", "Direction Z LINAC", 100, -1, 1);
    TH1F *bsGoodness = new TH1F("bsGoodness", "Bonsai Goodness LINAC",100, 0.4, 1.);
    TH1F *bsDirectionKS = new TH1F("bsDirectionKS", "Bonsai Direction KS LINAC",100, 0., 1.);
    Res res = {bsenergy, ovaq, r2z, Neff, Vertexes, Directions, bsGoodness, bsDirectionKS};
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
        bsovaq        = *reinterpret_cast<float*>(&LOWE->linfo[25]);// linfo=26: bonsai ovaq
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
            Neff->Fill(LOWE->bseffhit[0]);
            Vertexes[0]->Fill(LOWE->bsvertex[0]);
            Vertexes[1]->Fill(LOWE->bsvertex[1]);
            Vertexes[2]->Fill(LOWE->bsvertex[2]);
            Directions[0]->Fill(LOWE->bsdir[0]);
            Directions[1]->Fill(LOWE->bsdir[1]);
            Directions[2]->Fill(LOWE->bsdir[2]);
            bsGoodness->Fill(LOWE->bsgood[1]);
	    bsDirectionKS->Fill(LOWE->bsdirks);
        }

    }
    cout << "done." << endl;
    //file->Close();
    //cout<<bsenergy<<" "<< bsenergy->GetEntries()<<res.bsenergy<<" "<<res.bsenergy->GetEntries()<<endl;
    return res;
}
