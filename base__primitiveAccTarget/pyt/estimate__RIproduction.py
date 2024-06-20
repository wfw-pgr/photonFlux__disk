import os, sys, re, json, math
import numpy                      as np
import scipy.interpolate          as itp
import scipy.integrate            as itg
import scipy.optimize             as opt
import nkUtilities.plot1D         as pl1
import nkUtilities.load__config   as lcf
import nkUtilities.configSettings as cfs


# ========================================================= #
# ===  estimate__RIproduction.py                        === #
# ========================================================= #
def estimate__RIproduction():

    e_, pf_, xs_ =  0, 1, 1
    mb2cm2       =  1.0e-27
    paramsFile   =  "dat/parameters.json"
    
    # ------------------------------------------------- #
    # --- [1] load parameters from file             --- #
    # ------------------------------------------------- #
    import nkUtilities.json__formulaParser as jso
    params       = jso.json__formulaParser( inpFile=paramsFile )
    
    # ------------------------------------------------- #
    # --- [2] calculate parameters & define EAxis   --- #
    # ------------------------------------------------- #
    #  -- [2-1] energy axis                         --  #
    EAxis        = np.linspace( params["integral.EAxis.min"], params["integral.EAxis.max"], \
                                params["integral.EAxis.num"] )
    #  -- [2-2] calculate other parameters          --  #
    params       = calculate__parameters( params=params )

    # ------------------------------------------------- #
    # --- [3] load photon flux                      --- #
    # ------------------------------------------------- #
    pf_fit_uA, pf_raw = load__photonFlux( EAxis=EAxis, params=params )
    pf_fit            = params["photon.beam.current.use"] * pf_fit_uA
    
    # ------------------------------------------------- #
    # --- [4] load cross-section                    --- #
    # ------------------------------------------------- #
    import nkUtilities.load__pointFile as lpf
    xs_raw    = lpf.load__pointFile( inpFile=params["xsection.filename"], returnType="point")
    xs_fit_mb = fit__forRIproduction( xD=xs_raw[:,e_], yD=xs_raw[:,xs_], xI=EAxis, \
                                      mode=params["xsection.fit.method"], \
                                      p0=params["xsection.fit.p0"], \
                                      threshold=params["xsection.fit.Eth"] )
    xs_fit    = mb2cm2 * xs_fit_mb
    
    # ------------------------------------------------- #
    # --- [5] calculate dY(E)                       --- #
    # ------------------------------------------------- #
    dYield    = params["target.tN_product"] * pf_fit * xs_fit
    
    # ------------------------------------------------- #
    # --- [6] integrate dY(E) with respect to E     --- #
    # ------------------------------------------------- #
    if ( params["integral.method"] == "simpson" ):
        YieldRate = itg.simpson( dYield, x=EAxis )
    N_yield    = params["photon.beam.duration"] * 60*60.0  * YieldRate 
    A_yield    = params["product.lambda.1/s"]   * N_yield
    lam1,lam2  = params["product.lambda.1/s"], params["decayed.lambda.1/s"]
    t_max_s    = np.log( lam1/lam2 ) / ( lam1 - lam2 )
    t_max      = halflife__unitConvert( { "value":t_max_s, "unit":"s" }, to_unit="d" )["value"]
    ratio      = ( lam2/( lam2-lam1 ) )*( np.exp( -lam1*t_max_s ) - np.exp( -lam2*t_max_s ) )*100
    A_decay    = ( ratio/100.0 ) * A_yield
    efficiency = A_decay / ( params["target.activity.Bq"] * params["photon.beam.current.use"] \
                             * params["photon.beam.duration"] )
    results    = { "YieldRate":YieldRate, "N_yield":N_yield, "A_yield":A_yield, \
                   "t_max":t_max, "ratio":ratio, "A_decay":A_decay, "efficiency":efficiency }
    
    # ------------------------------------------------- #
    # --- [7] draw sigma(E), phi(E), dY(E)          --- #
    # ------------------------------------------------- #
    draw__figures( params=params, EAxis=EAxis, pf_fit=pf_fit_uA, pf_raw=pf_raw, \
                   xs_fit=xs_fit_mb, xs_raw=xs_raw, dYield=dYield )
    
    # ------------------------------------------------- #
    # --- [8] save & return                         --- #
    # ------------------------------------------------- #
    Data = { "params":params, "EAxis":EAxis, "results":results, \
             "pf_fit":pf_fit, "xs_fit":xs_fit, "dYield":dYield }
    write__results( Data=Data, params=params )
    return( YieldRate )


# ========================================================= #
# ===  load photon flux                                 === #
# ========================================================= #

def load__photonFlux( EAxis=None, params=None ):

    e_, f_             = 0, 1
    el_, eu_, pf_, er_ = 0, 1, 2, 3
    expr_from          = "^#\s*e\-lower"
    expr_to            = "^\s*$"
    
    # ------------------------------------------------- #
    # --- [1] load photon flux file                 --- #
    # ------------------------------------------------- #
    if   ( params["photon.filetype"] == "energy-fluence" ):
        import nkUtilities.load__pointFile as lpf
        pf_raw = lpf.load__pointFile( inpFile=params["photon.filename"], returnType="point" )
        pf_raw[:,f_] = pf_raw[:,f_] / params["photon.beam.current.sim"]
        
    elif ( params["photon.filetype"] == "phits-out"      ):
        # -- expr_from = r"^#\s*e\-lower",  expr_to = r"^\s*$"  -- #
        import nkUtilities.retrieveData__afterStatement as ras
        pf_ret = ras.retrieveData__afterStatement( inpFile  = params["photon.filename"], \
                                                   expr_from=expr_from, expr_to=expr_to )
        e_avg  =  0.5 * ( pf_ret[:,el_] + pf_ret[:,eu_] )
        p_nrm  = np.copy( pf_ret[:,pf_] ) / params["photon.beam.current.sim"]
        pf_raw = np.concatenate( [ e_avg[:,np.newaxis], p_nrm[:,np.newaxis] ], axis=1 )

    # ------------------------------------------------- #
    # --- [2] fit photon flux                       --- #
    # ------------------------------------------------- #
    pf_fit_uA  = fit__forRIproduction( xD=pf_raw[:,e_], yD=pf_raw[:,f_], \
                                       xI=EAxis, mode=params["photon.fit.method"], \
                                       p0=params["photon.fit.p0"], \
                                       threshold=params["photon.fit.Eth"] )
    return( pf_fit_uA, pf_raw )


# ========================================================= #
# ===  fit__forRIproduction                             === #
# ========================================================= #

def fit__forRIproduction( xD=None, yD=None, xI=None, mode="linear", p0=None, threshold=None ):

    # ------------------------------------------------- #
    # --- [1] fitting                               --- #
    # ------------------------------------------------- #
    if   ( mode == "linear"   ):
        fitFunc = itp.interp1d( xD, yD, kind="linear", fill_value="extrapolate" )
        yI      = fitFunc( xI )
    elif ( mode == "gaussian" ):
        fitFunc   = lambda eng,c1,c2,c3,c4,c5 : \
            c1*np.exp( -1.0/c2*( eng-c3 )**2 ) +c4*eng +c5
        copt,cvar = opt.curve_fit( fitFunc, xD, yD, p0=p0 )
        yI        = fitFunc( xI, *copt )
    else:
        print( "[estimate__RIproduction.py] undefined mode :: {} ".format( mode ) )
        sys.exit()
        
    # ------------------------------------------------- #
    # --- [2] threshold                             --- #
    # ------------------------------------------------- #
    if ( threshold is not None ):
        yI = np.where( xI > threshold, yI, 0.0 )
    return( yI )


# ========================================================= #
# ===  draw__figures                                    === #
# ========================================================= #
def draw__figures( params=None, EAxis=None, pf_fit=None, xs_fit=None, \
                   pf_raw=None, xs_raw=None, dYield=None ):

    min_, max_, num_ = 0, 1, 2
    e_, xs_, pf_     = 0, 1, 1

    # ------------------------------------------------- #
    # --- [1] configure data                        --- #
    # ------------------------------------------------- #
    if ( params["plot.norm.auto"] ):
        params["plot.dYield.norm"]   = 10.0**( math.floor( np.log10(abs(np.max(dYield))) ) )
        params["plot.photon.norm"]   = 10.0**( math.floor( np.log10(abs(np.max(pf_fit))) ) )
        params["plot.xsection.norm"] = 10.0**( math.floor( np.log10(abs(np.max(xs_fit))) ) )
        
    xs_fit_plot  = xs_fit        / params["plot.xsection.norm"]
    pf_fit_plot  = pf_fit        / params["plot.photon.norm"]
    xs_raw_plot  = xs_raw[:,xs_] / params["plot.xsection.norm"]
    pf_raw_plot  = pf_raw[:,pf_] / params["plot.photon.norm"]
    dY_plot      = dYield        / params["plot.dYield.norm"]
    xs_norm_str  = "10^{" + str( round( math.log10( params["plot.xsection.norm"] ) ) ) + "}"
    pf_norm_str  = "10^{" + str( round( math.log10( params["plot.photon.norm"]   ) ) ) + "}"
    dY_norm_str  = "10^{" + str( round( math.log10( params["plot.dYield.norm"]   ) ) ) + "}"
    label_xs_fit = "$\sigma_{fit}(E)/" + xs_norm_str + "\ \mathrm{(mb)}$"
    label_pf_fit = "$\phi_{fit}(E)/"   + pf_norm_str + "\ \mathrm{(photons/MeV/uA/s)}$"
    label_dY     = "$dY/ "             + dY_norm_str + "\ \mathrm{(atoms/MeV/s)}$"
    label_xs_raw = "$\sigma_{raw}(E)/" + xs_norm_str + "\ \mathrm{(mb)}$"
    label_pf_raw = "$\phi_{raw}(E)/"   + pf_norm_str + "\ \mathrm{(photons/MeV/uA/s)}$"
    
    # ------------------------------------------------- #
    # --- [2] configure plot                        --- #
    # ------------------------------------------------- #
    config                   = lcf.load__config()
    config                   = cfs.configSettings( configType="plot.def", config=config )
    config["FigSize"]        = (4.5,4.5)
    config["plt_position"]   = [ 0.16, 0.16, 0.94, 0.94 ]
    config["plt_xAutoRange"] = False
    config["plt_yAutoRange"] = False
    config["plt_xRange"]     = [ params["plot.xRange"][min_], params["plot.xRange"][max_] ]
    config["plt_yRange"]     = [ params["plot.yRange"][min_], params["plot.yRange"][max_] ]
    config["xMajor_Nticks"]  = int( params["plot.xRange"][num_] )
    config["yMajor_Nticks"]  = int( params["plot.yRange"][num_] )
    config["plt_marker"]     = "o"
    config["plt_markersize"] = 1.0
    config["plt_linestyle"]  = "-"
    config["plt_linewidth"]  = 1.2
    config["xTitle"]         = "Energy (MeV)"
    config["yTitle"]         = "$dY, \ \phi, \ \sigma$"

    # ------------------------------------------------- #
    # --- [3] plot                                  --- #
    # ------------------------------------------------- #
    fig     = pl1.plot1D( config=config, pngFile=params["plot.filename"] )
    fig.add__plot( xAxis=EAxis       , yAxis=dY_plot    , label=label_dY    , \
                   color="C0", marker="none"    )
    fig.add__plot( xAxis=EAxis       , yAxis=xs_fit_plot, label=label_xs_fit, \
                   color="C1", marker="none"    )
    fig.add__plot( xAxis=xs_raw[:,e_], yAxis=xs_raw_plot, label=label_xs_raw, \
                   color="C1", linestyle="none" )
    fig.add__plot( xAxis=EAxis       , yAxis=pf_fit_plot, label=label_pf_fit, \
                   color="C2", marker="none"    )
    fig.add__plot( xAxis=pf_raw[:,e_], yAxis=pf_raw_plot, label=label_pf_raw, \
                   color="C2", linestyle="none" )
    fig.add__legend( FontSize=9.0 )
    fig.set__axis()
    fig.save__figure()
    return()


# ========================================================= #
# ===  write__results                                   === #
# ========================================================= #
def write__results( Data=None, params=None, stdout="minimum" ):

    if ( Data is None ): sys.exit( "[estimate__RIproduction.py] Data == ???" )
    text1        = "[paramters]\n"
    text2        = "[results]\n"
    keys1        = [ "t_max", "ratio" ]
    keys2        = [ "YieldRate", "N_yield", "A_yield", "A_decay", "efficiency" ]
    paramsFormat = "{0:>30} :: {1}\n"
    resultFormat = "{0:>30} :: {1:15.8e}   {2}\n"
    resultUnits  = { "YieldRate":"(atoms/s)", "N_yield":"(atoms)", "A_yield":"(Bq)", \
                     "t_max":"(d)", "ratio":"(%)", "A_decay":"(Bq)", \
                     "efficiency":"(Bq(Ac)/(Bq(Ra) uA h))" }
        
    # ------------------------------------------------- #
    # --- [1] pack texts                            --- #
    # ------------------------------------------------- #
    for key,val in Data["params"].items():
        text1 += paramsFormat.format( key, val )
    text2 += "\n" + " "*3 + "-"*25 + " "*3 + "[ Decay Time ]" + " "*3 + "-"*25 + "\n"
    for key in keys1:
        text2 += resultFormat.format( key, Data["results"][key], resultUnits[key] )
    text2 += "\n" + " "*3 + "-"*25 + " "*3 + "[ Production ]" + " "*3 + "-"*25 + "\n"
    for key in keys2:
        text2 += resultFormat.format( key, Data["results"][key], resultUnits[key] )
    text2 += "\n" + " "*3 + "-"*70 + "\n"
    texts  = text1 + text2 + "\n"
    
    # ------------------------------------------------- #
    # --- [2] save and print texts                  --- #
    # ------------------------------------------------- #
    if   ( stdout == "all" ):
        print( texts )
    elif ( stdout == "minimum" ):
        print( "\n" + text2 + "\n" )

    if ( params["results.summaryFile"] is not None ):
        with open( params["results.summaryFile"], "w" ) as f:
            f.write( texts )
        print( "[estimate__RIproduction.py] summary    is saved in {}"\
               .format( params["results.summaryFile"] ) )

    if ( params["results.yieldFile"] is not None ):
        import nkUtilities.save__pointFile as spf
        Data_ = [ Data["EAxis"][:,np.newaxis] , Data["dYield"][:,np.newaxis],\
                  Data["pf_fit"][:,np.newaxis], Data["xs_fit"][:,np.newaxis] ]
        Data_ = np.concatenate( Data_, axis=1 )
        names = [ "energy(MeV)", "dYield(atoms/MeV/s)", \
                  "photonFlux(photons/MeV/uA/s)", "crossSection(mb)" ]
        spf.save__pointFile( outFile=params["results.yieldFile"], Data=Data_, silent=True )
        print( "[estimate__RIproduction.py] yield data is saved in {}"\
               .format( params["results.yieldFile"] ) )

    
# ========================================================= #
# ===  convert halflife's unit in seconds               === #
# ========================================================= #
def halflife__unitConvert( halflife, to_unit=None ):
    timeUnitDict  = { "s":1.0, "m":60.0, "h":60*60.0, \
                      "d":24*60*60.0, "y":365*24*60*60.0 }
    halflife  = { "value": halflife["value"]*timeUnitDict[ halflife["unit"] ], "unit":"s" }
    if ( to_unit is not None ):
        halflife  = { "value": halflife["value"]/timeUnitDict[ to_unit ], "unit":to_unit }
    return( halflife )


# ========================================================= #
# ===  calculate__parameters                            === #
# ========================================================= #
def calculate__parameters( params=None ):

    N_Avogadro   = 6.02e23
    mm2cm        = 0.1
    
    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( params is None ): sys.exit( "[estimate__RIproduction.py] params == ???" )

    # ------------------------------------------------- #
    # --- [2] calculate half-life time & lambda     --- #
    # ------------------------------------------------- #
    keys   = [ "target", "product", "decayed" ]
    for key in keys:
        key_h         = "{}.halflife"  .format( key )
        key_l         = "{}.lambda.1/s".format( key )
        params[key_h] = halflife__unitConvert( params[key_h] )
        params[key_l] = np.log(2.0) / params[key_h]["value"]
        
    # ------------------------------------------------- #
    # --- [3] volume & area model                   --- #
    # ------------------------------------------------- #
    if   ( params["target.area.type"].lower() == "direct" ):
        params["target.area.cm2"] = params["target.area.direct.cm2"]
    elif ( params["target.area.type"].lower() == "disk"   ):
        params["target.area.cm2"] = 0.25*np.pi*( mm2cm*params["target.area.diameter.mm"] )**2

    # ------------------------------------------------- #
    # --- [4] thickness x atom density              --- #
    # ------------------------------------------------- #
    params["target.atoms/cm3"] = N_Avogadro*( params["target.g/cm3"] / params["target.g/mol"] )
    if   ( params["target.thick.type"].lower() == "bq" ):
        N_atoms                     = params["target.activity.Bq"] / params["target.lambda.1/s"]
        V_target                    = N_atoms  / params["target.atoms/cm3"]
        params["target.thick.cm"]   = V_target / params["target.area.cm2"]
        params["target.tN_product"] = params["target.atoms/cm3"]*params["target.thick.cm"]
        
    elif ( params["target.thick.type"].lower() == "direct" ):
        params["target.thick.cm"]   = params["target.thick.direct.mm"] * mm2cm
        params["target.tN_product"] = params["target.atoms/cm3"]*params["target.thick.cm"]
        
    elif ( params["target.thick.type"].lower() == "fluence" ):
        N_atoms                     = params["target.activity.Bq"] / params["target.lambda.1/s"]
        V_target                    = N_atoms  / params["target.atoms/cm3"]
        params["target.thick.cm"]   = V_target / params["target.area.cm2"]
        params["target.tN_product"] = params["target.atoms/cm3"]
        #
        # thick t is included in :: photonFlux profile :: 
        # fluence = count/m2 = count*m/m3,   =>   fluence * volume = count*m
        # vol in phits's tally must be "V=1", or, flux will be devided by Volume V.
        #
    # ------------------------------------------------- #
    # --- [5] return                                --- #
    # ------------------------------------------------- #
    return( params )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #
if ( __name__=="__main__" ):
    estimate__RIproduction()
