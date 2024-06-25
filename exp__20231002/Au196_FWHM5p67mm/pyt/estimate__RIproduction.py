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

def estimate__RIproduction( paramsFile=None ):

    e_, pf_, xs_ =  0, 1, 1
    mb2cm2       =  1.0e-27
    
    # ------------------------------------------------- #
    # --- [1] load parameters from file             --- #
    # ------------------------------------------------- #
    if ( paramsFile is None ): sys.exit( "[estimate__RIproduction.py] paramsFile == ???" )
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
    xs_fit_mb, xs_raw = load__xsection  ( EAxis=EAxis, params=params )
    xs_fit            = mb2cm2 * xs_fit_mb
    
    # ------------------------------------------------- #
    # --- [5] calculate dY(E)                       --- #
    # ------------------------------------------------- #
    dYield    = params["target.tN_product"] * pf_fit * xs_fit
    
    # ------------------------------------------------- #
    # --- [6] integrate dY(E) with respect to E     --- #
    # ------------------------------------------------- #
    results   = integrate__yield( EAxis=EAxis, dYield=dYield, params=params )
    
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
    return( results["YieldRate"] )


# ========================================================= #
# ===  load photon flux                                 === #
# ========================================================= #

def load__photonFlux( EAxis=None, params=None ):
    
    e_, f_             = 0, 1
    el_, eu_, pf_, er_ = 0, 1, 2, 3
    expr_from          = "^#\s*e\-lower"
    expr_to            = "^\s*$"
    # -- (unit):: E: [MeV], phi: [photons m/MeV/s] -- #
    
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
        if   ( params["photon.bin2point.convert"] == "center" ):
            e_avg  =  0.5 * ( pf_ret[:,el_] + pf_ret[:,eu_] )
            p_nrm  = np.copy( pf_ret[:,pf_] ) / params["photon.beam.current.sim"]
        elif ( params["photon.bin2point.convert"] == "edge"   ):
            e_avg  = np.insert( pf_ret[:,el_], pf_ret.shape[0], pf_ret[-1,eu_] )
            p_nrm  = ( 0.5 * ( pf_ret[:,pf_] + np.roll( pf_ret[:,pf_], +1 ) ) )[1:]
            p_nrm  = np.insert( p_nrm, [ 0, p_nrm.shape[0] ], [ pf_ret[0,pf_], pf_ret[-1,pf_] ] )
            p_nrm  = p_nrm / params["photon.beam.current.sim"]
        else:
            print( "[estimate__RIproduction.py] unknown photon.bin2point.convert [ERROR] " )
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
# ===  load x-section Data                              === #
# ========================================================= #

def load__xsection( EAxis=None, params=None ):
    
    e_, xs_  = 0, 1
    eV2MeV   = 1.0e-6
    b2mb     = 1.0e+3

    # ------------------------------------------------- #
    # --- [1] load data                             --- #
    # ------------------------------------------------- #
    import nkUtilities.load__pointFile as lpf
    xs_raw     = lpf.load__pointFile( inpFile=params["xsection.filename"], returnType="point")
    if   ( params["xsection.database"].lower() == "jendl" ):
        xs_raw[:, e_] = xs_raw[:, e_] * eV2MeV
        xs_raw[:,xs_] = xs_raw[:,xs_] * b2mb
    elif ( params["xsection.database"].lower() == "tendl" ):
        pass
    else:
        print( "[estimate__RIproduction.py] xsection.database == {} ??? "\
               .format( params["xsection.database"] ) )
    
    # ------------------------------------------------- #
    # --- [2] fit x-section Data                    --- #
    # ------------------------------------------------- #
    xs_fit_mb = fit__forRIproduction( xD=xs_raw[:,e_], yD=xs_raw[:,xs_], xI=EAxis, \
                                      mode=params["xsection.fit.method"], \
                                      p0=params["xsection.fit.p0"], \
                                      threshold=params["xsection.fit.Eth"] )
    return( xs_fit_mb, xs_raw )


# ========================================================= #
# ===  integrate__yield                                 === #
# ========================================================= #

def integrate__yield( EAxis=None, dYield=None, params=None ):
    
    # ------------------------------------------------- #
    # --- [1] integrate along EAxis                 --- #
    # ------------------------------------------------- #
    if   ( params["integral.method"] == "simpson"     ):
        YieldRate = itg.simpson  ( dYield, x=EAxis )
    elif ( params["integral.method"] == "trapezoid"   ):
        YieldRate = itg.trapezoid( dYield, x=EAxis )
    elif ( params["integral.method"] == "rectangular" ):
        YieldRate = np.dot( np.diff( EAxis ), dYield[:-1] )
    else:
        print( "[estimate__RIproduction.py] integral.method == {} ??? "\
               .format( params["integral.method"] ) )
        sys.exit()

    # ------------------------------------------------- #
    # --- [2] calculate results                     --- #
    # ------------------------------------------------- #
    #  -- Taylor (linear) case :: N_yield = params["photon.beam.duration"]*60*60 * YieldRate
    #  -- Non-Linear case      :: N_yield = Y0/L [ 1 - exp( - L t ) ]  -- #
    lambda_t  = params["product.lambda.1/s"]   * ( params["photon.beam.duration"] * 60*60.0 )
    N_max     = YieldRate / params["product.lambda.1/s"]
    N_yield   = N_max * ( 1.0 - np.exp( -1.0*lambda_t ) )
    A_yield   = params["product.lambda.1/s"]   * N_yield

    # ------------------------------------------------- #
    # --- [3] predict decay production              --- #
    # ------------------------------------------------- #
    if ( params["decayed.halflife"] is not None ):
        lam1,lam2 = params["product.lambda.1/s"], params["decayed.lambda.1/s"]
        t_max_s   = np.log( lam1/lam2 ) / ( lam1 - lam2 )
        t_max     = halflife__unitConvert( {"value":t_max_s,"unit":"s"}, to_unit="d" )["value"]
        ratio     = ( lam2/(lam2-lam1) )*( np.exp( -lam1*t_max_s )-np.exp( -lam2*t_max_s ) )*100
        A_decay   = ( ratio/100.0 )* A_yield
    else:
        t_max, ratio, A_decay = None, None, None

    # ------------------------------------------------- #
    # --- [4] calculate efficiency                  --- #
    # ------------------------------------------------- #
    eta_yield_Bq, eta_decay_Bq, eta_decay_wt = None, None, None
    charge       = params["photon.beam.current.use"] * params["photon.beam.duration"]
    target_Bq    = params["target.activity.Bq"]
    eta_yield_wt = A_yield / ( params["target.mass.mg"] * charge )
    if ( ( target_Bq is not None ) ):
        eta_yield_Bq = A_yield / ( target_Bq * charge )
    if ( ( target_Bq is not None ) and ( A_decay is not None ) ):
        eta_decay_Bq = A_decay / ( target_Bq * charge )
    if (                               ( A_decay is not None ) ):
        eta_decay_wt = A_decay / ( params["target.mass.mg"] * charge )

    # ------------------------------------------------- #
    # --- [5] return results                        --- #
    # ------------------------------------------------- #
    results   = { "YieldRate":YieldRate, "N_yield":N_yield, "A_yield":A_yield, \
                  "t_max":t_max, "ratio":ratio, "A_decay":A_decay, \
                  "eta_yield_Bq":eta_yield_Bq, "eta_yield_wt":eta_yield_wt, \
                  "eta_decay_Bq":eta_decay_Bq, "eta_decay_wt":eta_decay_wt, }
    return( results )

    
# ========================================================= #
# ===  fit__forRIproduction                             === #
# ========================================================= #

def fit__forRIproduction( xD=None, yD=None, xI=None, mode="linear", p0=None, threshold=None ):

    # ------------------------------------------------- #
    # --- [1] fitting                               --- #
    # ------------------------------------------------- #
    if   ( mode == "linear"   ):
        fitFunc   = itp.interp1d( xD, yD, kind="linear", fill_value="extrapolate" )
        yI        = fitFunc( xI )
    elif ( mode == "gaussian" ):
        fitFunc   = lambda eng,c1,c2,c3,c4,c5 : \
            c1*np.exp( -1.0/c2*( eng-c3 )**2 ) +c4*eng +c5
        copt,cvar = opt.curve_fit( fitFunc, xD, yD, p0=p0 )
        yI        = fitFunc( xI, *copt )
    elif ( mode == "log-poly5th" ):
        indx      = np.where( ( xD >= np.min(xI) ) & ( xD <= np.max(xI) ) & ( yD > 0.0 ) )
        xT, yT    = xD[ indx ], np.log10( yD[indx] )
        xlims     = [ xT[0], xT[-1] ]
        fitFunc   = lambda eng,c0,c1,c2,c3,c4,c5,c6 : \
            c0 + c1*eng + c2*eng**2 + c3*eng**3 + c4*eng**4 + c5*eng**5 + c6*eng**6
        copt,cvar = opt.curve_fit( fitFunc, xT, yT, p0=p0 )
        yI        = np.where( ( xI >= xlims[0] ) & ( xI <= xlims[1] ), \
                              10.0**( fitFunc( xI, *copt ) ), 0.0 )
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
    keysdict     = { "product": [ "YieldRate", "N_yield", "A_yield", \
                                  "eta_yield_wt", "eta_yield_Bq" ],\
                     "decayed": [ "t_max",   "ratio", "A_decay", \
                                  "eta_decay_wt", "eta_decay_Bq" ]
    }
    titlesFormat = "\n"+" "*3+"-"*25+" "*3 + "{}" + " "*3+"-"*25+" "*3 + "\n\n"
    paramsFormat = "{0:>30} :: {1}\n"
    resultFormat = "{0:>30} :: {1:15.8e}   {2}\n"
    none__Format = "{0:>30} :: {1:}   {2}\n"
    resultUnits  = { "YieldRate":"(atoms/s)", "N_yield":"(atoms)", "A_yield":"(Bq)", \
                     "t_max":"(d)", "ratio":"(%)", "A_decay":"(Bq)", \
                     "eta_yield_wt":"(Bq/(mg uA h))", "eta_decay_wt":"(Bq/(mg uA h))", \
                     "eta_yield_Bq":"(Bq/(Bq uA h))", "eta_decay_Bq":"(Bq/(Bq uA h))" }
    
    # ------------------------------------------------- #
    # --- [1] pack texts                            --- #
    # ------------------------------------------------- #
    results = Data["results"]
    for key,val in Data["params"].items():
        text1 += paramsFormat.format( key, val )

    for label in keysdict.keys():
        text2 += titlesFormat.format( "[ {} ]".format( label.center(10) ) )
        for key in keysdict[label]:
            if ( results[key] is not None ):
                text2 += resultFormat.format( key, results[key], resultUnits[key] )
            else:
                text2 += none__Format.format( key, results[key], resultUnits[key] )
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
    mg2g         = 1.0e-3
    
    # ------------------------------------------------- #
    # --- [1] arguments                             --- #
    # ------------------------------------------------- #
    if ( params is None ): sys.exit( "[estimate__RIproduction.py] params == ???" )

    # ------------------------------------------------- #
    # --- [2] calculate half-life time & lambda     --- #
    # ------------------------------------------------- #
    keys = [ "target", "product", "decayed" ]
    for key in keys:
        key_h = "{}.halflife"  .format( key )
        key_l = "{}.lambda.1/s".format( key )
        if ( params[key_h]["value"] is not None ):
            params[key_h] = halflife__unitConvert( params[key_h] )
            params[key_l] = np.log(2.0) / params[key_h]["value"]
        else:
            params[key_h], params[key_l] = None, None
    if ( params["product.halflife"] is None ):
        print( "[estimate__RIproduction.py]  [ERROR] product must be RI, at least." )
        print( "[estimate__RIproduction.py]  product.halflife == {}"\
               .format( params["product.halflife"] ) )
        sys.exit()
        
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
        
    elif ( params["target.thick.type"].lower() == "fluence-bq" ):
        N_atoms                     = params["target.activity.Bq"]/params["target.lambda.1/s"]
        V_target                    = N_atoms  / params["target.atoms/cm3"]
        params["target.thick.cm"]   = V_target / params["target.area.cm2"]
        params["target.tN_product"] = params["target.atoms/cm3"]
        params["target.mass.mg"]    = 1.0
        
    elif ( params["target.thick.type"].lower() == "fluence-mass" ):
        V_target                    = ( params["target.mass.mg"]*mg2g ) / params["target.g/cm3"]
        params["target.thick.cm"]   = V_target / (params["target.area.cm2"])
        params["target.tN_product"] = params["target.atoms/cm3"]
        #
        # thick t is included in :: photonFlux profile :: 
        # fluence = count/m2 = count*m/m3,   =>   fluence * volume = count*m
        # vol in phits's tally must be "V=1", or, flux will be devided by Volume V.
        #
    else:
        print( "[estimate__RIproduction.py] target.thick.type == {} ?? [ERROR] "\
               .format( params["target.thick.type"]  ) )
        sys.exit()
    # ------------------------------------------------- #
    # --- [5] return                                --- #
    # ------------------------------------------------- #
    return( params )


# ========================================================= #
# ===   Execution of Pragram                            === #
# ========================================================= #
if ( __name__=="__main__" ):

    # ------------------------------------------------- #
    # --- [1] argument                              --- #
    # ------------------------------------------------- #
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( "--paramsFile", help="input file name." )
    args   = parser.parse_args()
    if ( args.paramsFile is None ):
        args.paramsFile = "dat/ri_prod.json"
        print( "[estimate__RIproduction.py] paramsFile (default) == {}".format(args.paramsFile))
    if ( not( os.path.exists( args.paramsFile ) ) ):
        print( "[estimate__RIproduction.py] paramsFile does not exists [ERROR]" )
        print( "[estimate__RIproduction.py] (e.g.) python pyt/estimate__RIproduction.py --paramsFile ri_prod.json" )
        sys.exit()

    # ------------------------------------------------- #
    # --- [2] call Main Routine                     --- #
    # ------------------------------------------------- #
    estimate__RIproduction( paramsFile=args.paramsFile )
