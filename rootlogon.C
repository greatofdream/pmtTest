{
printf("\nLoad skofl library.\n");
gSystem->SetIncludePath(" -I$SKOFL_ROOT/include");

gSystem->Load("$SKOFL_ROOT/lib/libDataDefinition.so");
gSystem->Load("$SKOFL_ROOT/lib/libtqrealroot.so");
gSystem->Load("$SKOFL_ROOT/lib/libloweroot.so");
gSystem->Load("$SKOFL_ROOT/lib/libatmpdroot.so");
gSystem->Load("$SKOFL_ROOT/lib/libmcinfo.so");
gSystem->Load("$SKOFL_ROOT/lib/libsofttrgroot.so");

gROOT->SetStyle("Plain");

gStyle->SetOptDate(1);
//gStyle->SetOptStat(0);
gStyle->SetPadBorderSize(0.0); 
gStyle->SetTitleBorderSize(0.0);

//gStyle->SetStatBorderSize(1.0);
//gStyle->SetTitleFontSize(0.5);
//gStyle->SetLabelOffset(1.2);
//gStyle->SetLabelFont(72);
//gStyle->SetPadColor(2);
gStyle->SetPalette(1);  // use the nice red->blue palette
printf(" .rootlogon.C script finished.\n");

}

