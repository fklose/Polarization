import ROOT
from ROOT import TFile
import sys

in_path = sys.argv[1]

file = TFile(in_path, "READ")

variables = [
	"TTTL_OP_Beam[0]",
    "Run_time[0]",
	"TDC_PHOTO_DIODE_LE[0]",
	"TDC_ION_MCP_LE[0]",
	"TDC_ION_MCP_LE_Count[0]",
	"TDC_PHOTO_DIODE_LE_Count[0]",
	"TDC_DL_X1_LE[0]",
	"TDC_DL_X2_LE[0]",
	"TDC_DL_Z1_LE[0]",
	"TDC_DL_Z2_LE[0]",
	"TDC_DL_X1_LE_Count[0]",
	"TDC_DL_X2_LE_Count[0]",
	"TDC_DL_Z1_LE_Count[0]",
	"TDC_DL_Z2_LE_Count[0]",
	]

if int(in_path[-10:-5]) >= 3438:
	variables += [
        "QDC_EIO0[0]",
	    "QDC_EIO1[0]",
		"QDC_EIO2[0]",
		"QDC_EIO3[0]",
		"QDC_EIO4[0]",
		"QDC_EIO5[0]"
        ]

tup = file.Get("ntuple")
tup.SetScanField(0)
tup.Scan(":".join(variables))