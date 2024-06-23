import numpy                                    as np
import nkUtilities.retrieveData__afterStatement as ras
import nkUtilities.save__pointFile              as spf

# ========================================================= #
# ===  fluence_energy.py                                === #
# ========================================================= #

def fluence_energy( inpFile=None, outFile=None, pngFile=None ):

    el_, eu_, ph_, er_ = 0, 1, 2, 3
    ea_, fl_           = 0, 1
    
    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( inpFile is None ): sys.exit( "[fluence_energy.py] inpFile == ???" )
    if ( outFile is None ): outFile = "dat/fluence_energy_extracted.dat"
    if ( pngFile is None ): pngFile = "png/fluence_energy_extracted.png"
    
    # ------------------------------------------------- #
    # --- [2] retrieve data from output             --- #
    # ------------------------------------------------- #
    expr_from  = r"^#\s*e\-lower"
    expr_to    = r"^\s*$"
    Data       = ras.retrieveData__afterStatement( inpFile=inpFile, outFile=outFile, \
                                                   expr_from=expr_from, expr_to=expr_to )
    ave        = np.zeros( (Data.shape[0],2,) )
    ave[:,ea_] = 0.5 * ( Data[:,el_] + Data[:,eu_] )
    ave[:,fl_] = np.copy( Data[:,ph_] )
    import nkUtilities.save__pointFile as spf
    spf.save__pointFile( outFile=outFile.replace("_extracted","_phitsin"), Data=ave )

    
    # ------------------------------------------------- #
    # --- [3] plot 1d graph                         --- #
    # ------------------------------------------------- #
    import nkUtilities.plot1D         as pl1
    import nkUtilities.load__config   as lcf
    import nkUtilities.configSettings as cfs
    x_,y_                    = 0, 1
    config                   = lcf.load__config()
    config                   = cfs.configSettings( configType="plot.def", config=config )
    config["FigSize"]        = (4.5,4.5)
    config["plt_position"]   = [ 0.16, 0.16, 0.94, 0.94 ]
    config["plt_xAutoRange"] = True
    config["plt_yAutoRange"] = False
    config["plt_xRange"]     = [ -1.2, +1.2 ]
    config["plt_yRange"]     = [ -1.2, +1.2 ]
    config["xMajor_Nticks"]  = 11
    config["yMajor_Nticks"]  = 11
    config["plt_marker"]     = "o"
    config["plt_markersize"] = 3.0
    config["plt_linestyle"]  = "-"
    config["plt_linewidth"]  = 2.0
    config["xTitle"]         = "energy (MeV)"
    config["yTitle"]         = "fluence (source/MeV/" + "$\mathrm{cm^{2}}$" + ")"

    # ------------------------------------------------- #
    # --- [4] plot 1d linear scale                  --- #
    # ------------------------------------------------- #
    pngFile1             = pngFile.replace( ".png", "_lin.png" )
    config["plt_ylog"]   = False
    config["plt_yRange"] = [ 0.e0, 2.e8 ]
    fig                  = pl1.plot1D( config=config, pngFile=pngFile1 )
    fig.add__plot( xAxis=Data[:,el_], yAxis=Data[:,ph_] )
    fig.set__axis()
    fig.save__figure()

    # ------------------------------------------------- #
    # --- [5] plot 1d log scale                     --- #
    # ------------------------------------------------- #
    pngFile2             = pngFile.replace( ".png", "_log.png" )
    config["plt_ylog"]   = True
    config["plt_yRange"] = [ 1.e2, 1.e10 ]
    fig                  = pl1.plot1D( config=config, pngFile=pngFile2 )
    fig.add__plot( xAxis=Data[:,el_], yAxis=Data[:,ph_] )
    fig.set__axis()
    fig.save__figure()
    return()


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #

if ( __name__=="__main__" ):

    inpFile = "out/fluence_energy.dat"
    fluence_energy( inpFile=inpFile )

    
