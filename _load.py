import ROOT
import numpy as np

def load_data(file_path : str) -> dict:
    """Loads relevant polarization data from the .root file at file_path

    Args:
        file_path (str): path where .root file is located

    Returns:
        dict: Dictionary containing numpy arrays of all necessary observables
    """
    
    file = ROOT.TFile(file_path)
    ntuple = file.Get("ntuple")
    
    EVENT = []
    QDC_EIO0, QDC_EIO1, QDC_EIO2, QDC_EIO3, QDC_EIO4, QDC_EIO5 = [], [], [], [], [], []
    TTTL_OP_Beam = []
    TDC_PHOTO_DIODE, TDC_ION_MCP = [], []
    TDC_DL_X1, TDC_DL_X2 = [], []
    TDC_DL_Z1, TDC_DL_Z2 = [], []
    
    for n, entry in enumerate(ntuple):
        EVENT.append(n)
        
        QDC_EIO0.append(entry.QDC_EIO0)
        QDC_EIO1.append(entry.QDC_EIO1)
        QDC_EIO2.append(entry.QDC_EIO2)
        QDC_EIO3.append(entry.QDC_EIO3)
        QDC_EIO4.append(entry.QDC_EIO4)
        QDC_EIO5.append(entry.QDC_EIO5)
        
        TTTL_OP_Beam.append(entry.TTTL_OP_Beam)
        
        # Track TDC data. Note that TDC data is given as a vector which does not always contain a number
        # hence we need to check that there is at least one entry in each TDC vector before trying to grab
        # it otherwise the code will not work.
        # If a TDC vector does not contain an element add np.nan to the array instead this allows for the use 
        # of array operations (np.nan does not).
        if entry.TDC_PHOTO_DIODE_LE_Count > 0:
            TDC_PHOTO_DIODE.append(entry.TDC_PHOTO_DIODE_LE[0])
        else:
            TDC_PHOTO_DIODE.append(np.nan)
    
        if entry.TDC_ION_MCP_LE_Count > 0:
            TDC_ION_MCP.append(entry.TDC_ION_MCP_LE[0])
        else:
            TDC_ION_MCP.append(np.nan)
        
        if entry.TDC_DL_X1_LE_Count > 0:
            TDC_DL_X1.append(entry.TDC_DL_X1_LE[0])
        else:
            TDC_DL_X1.append(np.nan)    
        
        if entry.TDC_DL_X2_LE_Count > 0:
            TDC_DL_X2.append(entry.TDC_DL_X2_LE[0])
        else:
            TDC_DL_X2.append(np.nan)    
            
        if entry.TDC_DL_Z1_LE_Count > 0:
            TDC_DL_Z1.append(entry.TDC_DL_Z1_LE[0])
        else:
            TDC_DL_Z1.append(np.nan)
            
        if entry.TDC_DL_Z2_LE_Count > 0:
            TDC_DL_Z2.append(entry.TDC_DL_Z2_LE[0])
        else:
            TDC_DL_Z2.append(np.nan)
            
    # Define conversion factor to properly convert TDC data.
    # According to OnlineHistograms.cpp in rootana src code this value
    # comes from 'Equipment/VME/Settings/V1192_ns_conversion' and seems
    # to be a hardware specific value.
    v1192_to_ns = 0.098765625
    
    # Make dictionary containing numpy arrays of all observables also take care of scaling
    dict = {
        "EVENT"             : np.asarray(EVENT),
        "TTTL_OP_Beam"      : np.asarray(TTTL_OP_Beam) / 1e6,               # us
        "TDC_PHOTO_DIODE"   : np.asarray(TDC_PHOTO_DIODE) * v1192_to_ns,    # ns
        "TDC_ION_MCP"       : np.asarray(TDC_ION_MCP) * v1192_to_ns,        # ns
        "TDC_DL_X1"         : np.asarray(TDC_DL_X1) * v1192_to_ns,          # ns
        "TDC_DL_X2"         : np.asarray(TDC_DL_X2) * v1192_to_ns,          # ns
        "TDC_DL_Z1"         : np.asarray(TDC_DL_Z1) * v1192_to_ns,          # ns
        "TDC_DL_Z2"         : np.asarray(TDC_DL_Z2) * v1192_to_ns,          # ns
        "QDC_EIO0"          : np.asarray(QDC_EIO0),
        "QDC_EIO1"          : np.asarray(QDC_EIO1),
        "QDC_EIO2"          : np.asarray(QDC_EIO2),
        "QDC_EIO3"          : np.asarray(QDC_EIO3),
        "QDC_EIO4"          : np.asarray(QDC_EIO4),
        "QDC_EIO5"          : np.asarray(QDC_EIO5)
    }
    
    # THRESHOLD is used to convert QDC_EIO data into true (1) or false (0) logic levels
    THRESHOLD = 1000
    
    # Compute extra observables
    observables = {
        "X"     : dict["TDC_DL_X1"] - dict["TDC_DL_X2"],
        "Y"     : dict["TDC_ION_MCP"] - dict["TDC_PHOTO_DIODE"],
        "Z"     : dict["TDC_DL_Z1"] - dict["TDC_DL_Z2"],
        "BITS"  : 1  * (dict["QDC_EIO0"] > THRESHOLD) \
                + 2  * (dict["QDC_EIO1"] > THRESHOLD) \
                + 4  * (dict["QDC_EIO2"] > THRESHOLD) \
                + 8  * (dict["QDC_EIO3"] > THRESHOLD) \
                + 16 * (dict["QDC_EIO4"] > THRESHOLD) \
                + 32 * (dict["QDC_EIO5"] > THRESHOLD) \
    }
    
    dict.update(observables)
    
    return dict


def generate_histograms(data : dict, cuts : dict, fname="./histograms.root") -> None:
    """Instead of making images, makes a .root file containing the relevant histograms.
    This is much faster than rendering the histograms with matplotlib and is recommended.
    """
    
    # Create output .root file
    output = ROOT.TFile.Open(fname, "RECREATE")
    
    # Make histogram objects
    X = ROOT.TH1D("TDC_DL_X1_LE[0] - TDC_DL_X2_LE[0]", "", 1000, -100.5, 100.5)
    Y = ROOT.TH1D("TDC_ION_MCP_LE[0] - TDC_PHOTO_DIODE_LE[0]", "", 100000, -10000.5, 10000.5)
    Z = ROOT.TH1D("TDC_DL_Z1_LE[0] - TDC_DL_Z2_LE[0]", "", 1000, -100.5, 100.5)
    TTTL_OP_Beam = ROOT.TH1D("TTTL_OP_Beam", "", 420, 0, 4200)
    DelayLineAnode_2d = ROOT.TH2D("DelayLineAnode-2d", "", 1000, -100.5, 100.5, 1000, -100.5, 100.5)
    
    # Add axis labels, title, etc.
    X.GetXaxis().SetTitle("TDC_DL_X1_LE[0] - TDC_DL_X2_LE[0] (ns)")
    X.GetYaxis().SetTitle("Counts")
    
    Y.GetXaxis().SetTitle("TDC_ION_MCP_LE[0] - TDC_PHOTO_DIODE_LE[0] (ns)")
    Y.GetYaxis().SetTitle("Counts")
    
    Z.GetXaxis().SetTitle("TDC_DL_Z1_LE[0] - TDC_DL_Z2_LE[0] (ns)")
    Z.GetYaxis().SetTitle("Counts")
    
    TTTL_OP_Beam.GetXaxis().SetTitle("Time (#mus)")
    TTTL_OP_Beam.GetYaxis().SetTitle("Counts")
    
    DelayLineAnode_2d.GetXaxis().SetTitle("TDC_DL_X1_LE[0] - TDC_DL_X2_LE[0] (ns)")
    DelayLineAnode_2d.GetYaxis().SetTitle("TDC_DL_Z1_LE[0] - TDC_DL_Z2_LE[0] (ns)")
    DelayLineAnode_2d.SetOption("COLZ")
    
    # Copy objects so that I can make histograms with and without cuts
    X_CUTS = X.Clone()
    Y_CUTS = Y.Clone()
    Z_CUTS = Z.Clone()
    TTTL_OP_Beam_CUTS = TTTL_OP_Beam.Clone()
    DelayLineAnode_2d_CUTS = DelayLineAnode_2d.Clone()
    
    # Crop the cut histograms
    X_CUTS.GetXaxis().SetRangeUser(*cuts["X"])
    Y_CUTS.GetXaxis().SetRangeUser(*cuts["Y"])
    Z_CUTS.GetXaxis().SetRangeUser(*cuts["Z"])
    TTTL_OP_Beam_CUTS.GetXaxis().SetRangeUser(*cuts["TTTL_OP_Beam"])
    DelayLineAnode_2d_CUTS.GetXaxis().SetRangeUser(*cuts["X"])
    DelayLineAnode_2d_CUTS.GetYaxis().SetRangeUser(*cuts["Z"])
    
    for x, y, z, tttl_op_beam in zip(data["X"], data["Y"], data["Z"], data["TTTL_OP_Beam"]):
        X.Fill(x)
        Y.Fill(y)
        Z.Fill(z)
        TTTL_OP_Beam.Fill(tttl_op_beam)
        DelayLineAnode_2d.Fill(x, z)
        
        X_CUT = (cuts["X"][0] <= x) & (x <= cuts["X"][1])
        Y_CUT = (cuts["Y"][0] <= y) & (y <= cuts["Y"][1])
        Z_CUT = (cuts["Z"][0] <= z) & (z <= cuts["Z"][1])
        TTTL_OP_Beam_CUT = (cuts["TTTL_OP_Beam"][0] <= tttl_op_beam) & (tttl_op_beam <= cuts["TTTL_OP_Beam"][1])
        
        if X_CUT:
            X_CUTS.Fill(x)
        
        if Y_CUT:
            Y_CUTS.Fill(y)
        
        if Z_CUT:
            Z_CUTS.Fill(z)
            
        if TTTL_OP_Beam_CUT:
            TTTL_OP_Beam_CUTS.Fill(tttl_op_beam)
        
        if (X_CUT & Z_CUT):
            DelayLineAnode_2d_CUTS.Fill(x, z)
    
    output.mkdir("Original Data")
    output.mkdir("Cut Data")
    
    output.cd("Original Data")    
    X.Write()
    Y.Write()
    Z.Write()
    TTTL_OP_Beam.Write()
    DelayLineAnode_2d.Write()
    
    output.cd("Cut Data")
    X_CUTS.Write()
    Y_CUTS.Write()
    Z_CUTS.Write()
    TTTL_OP_Beam_CUTS.Write()
    DelayLineAnode_2d_CUTS.Write()
    
    output.Close()
    
    return