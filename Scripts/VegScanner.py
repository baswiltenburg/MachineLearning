"""
Script Name: Vegetation scanner BETA
Description: Export MXD to Fishnet based vegetation scanner result. Performed in slices to process wms services
Created By: Len Geisler
Date: 04/08/2017
Changed by Sjoerd Hoekstra
Date: 27/09/2017
"""
# https://gis.stackexchange.com/questions/119503/getting-arcpy-to-work-with-anaconda
# import os; d = r"C:\Program Files\ArcGIS\Pro\bin"; os.environ["PATH"] = r"{};{}".format(d, os.environ["PATH"])
# import sys; d = r"C:\Program Files\ArcGIS\Pro\bin"; sys.path.append(d)
# import sys; d = r"C:\Program Files\ArcGIS\Pro\Resources\ArcPy"; sys.path.append(d)
# import sys; d = r"C:\Program Files\ArcGIS\Pro\Resources\ArcToolbox\Scripts"; sys.path.append(d)
# import sys; d = r"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3"; sys.path.append(d)
# import sys; d = r"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\Lib\site-packages"; sys.path.append(d)
# import os; e = "FOR_DISABLE_CONSOLE_CTRL_HANDLER"; os.environ[e] = '1' if (not e in os.environ) else ""

import arcpy, os
from arcpy.sa import *
from arcpy    import CopyRaster_management as CopyRaster

from common   import BaseTool
from common   import parameter

class VegetatieScanner(BaseTool):

    def __init__(self):
        """
        Initialization.
        """
        self.label = '5. VegetatieScanner'
        self.description = 'Exports an area in an MXD to a georeferenced vegetation scanner result. ' \
                           'Performed in slices to give the capability to extract WMS services both aerial picture as AHN'
        self.canRunInBackground = False

        # Set path to Generic modules
        import inspect, os, sys
        dirThisScript = os.path.dirname(inspect.getfile(inspect.currentframe()))
        dirLevelUp = os.path.abspath(os.path.join(dirThisScript, '..'))
        ##        dirGeneric = os.path.join(dirLevelUp,'Generic')
        dirGeneric = sys.path.append(r'c:\SD\TauwToolbox\Generic')
        sys.path.append(dirGeneric)

        # Import Tauw modules
        import clsGeneralUse
        #self.objTT = TT_GeneralUse.TT_GeneralUse(sys, arcpy) # Voor productie, met minimale logging
        self.objTT = clsGeneralUse.TT_GeneralUse(sys, arcpy, debug=False, logPrint=True) # voor logging met objTT.AddDebug en print

        '''Create your parameters here using the paramater function.
        Make sure you leave the enclosing brackets and separate your
        parameters using commas.
        parameter(displayName, name, datatype, defaultValue=None, parameterType='Required', direction='Input')
        '''


    def getParameterInfo(self):
        """ return Parameter definitions."""
        self.parameters =   [
                            parameter('AHNi (maaiveld)'              ,'AHNi'     ,'DERasterDataset'),
                            parameter('AHNr (ruw)'                   ,'AHNr'     ,'DERasterDataset'),
                            parameter('CIR luchtfoto'                ,'lufo'     ,'DERasterDataset'),
                            parameter('Output gdb (wordt aangemaakt)','outputgdb','DEWorkspace'    ,direction="Output")
                            ]
        self.parameters[3].value = "C:\GIS\VegScan.gdb"

        return self.parameters

    def updateMessages(self, parameters):
        """
        This example overrides the implementation in BaseTool, but still uses
        the required functionality for maintaining a reference to the tool's
        parameters.
        """
        if parameters[3].altered and parameters[3].valueAsText.upper()[-4:] !='.GDB':
            parameters[3].value = parameters[3].valueAsText + ".gdb"

        super(VegetatieScanner, self).updateMessages(parameters)

	def isLicensed(self):
		"""Allow the tool to execute, only if the ArcGIS Spatial Analyst extension
		is available."""
		try:
			if arcpy.CheckExtension("Spatial") != "Available":
				raise Exception
		except Exception:
			return False  # tool cannot be executed

		return True  # tool can be executed


    def execute(self, parameters, messages):
        """
        Executes the tool.
        """

        try:
            self.objTT.StartAnalyse()
            
            # Variabelen
            AHN_i     =     parameters[0].valueAsText 
            AHN_r     =     parameters[1].valueAsText 
            lufo      =     parameters[2].valueAsText
            outputgdb =     parameters[3].valueAsText

            #Check out the Spatial Analyst extension (must be available to check out!)
            arcpy.CheckOutExtension("Spatial")
            arcpy.env.overwriteOutput = True

            # Aanmaken output gdb
            if not arcpy.Exists(outputgdb):
                arcpy.CreateFileGDB_management(os.path.dirname(outputgdb), os.path.basename(outputgdb))
                self.objTT.AddMessage("GDB wordt aangemaakt!")
            else:
                self.objTT.AddMessage("GDB bestaat al!")

            arcpy.env.workspace = outputgdb

            #Folder moet dynamische input. 
            self.objTT.AddMessage("Luchtfoto in kleur banden splitsen")
            NIR = Raster(CopyRaster(lufo+r"\Band_1", "NIR"))
            Red = Raster(CopyRaster(lufo+r"\Band_2", "Red"))

            self.objTT.AddMessage("NDVI berekenen")
            Num     = Float(NIR - Red)
            Denom   = Float(NIR + Red)
            NDVI_eq = Divide(Num, Denom)
            NDVI_eq.save('NDVI') #Saving output to result output you specified above

            #Vanaf hier wordt vegetatie kaart gegenereerd
            self.objTT.AddMessage("Vegetatie bepalen uit NDVI waarden")
            Vegetatie = Con(NDVI_eq>0.05,1,0)

            #Creatie AHN info
            VegetatieAHN = SetNull(Vegetatie,AHN_r,"Value = 0")
            VegetatieAHN.save("Veg_AHN")

            HoogteVegetatie = VegetatieAHN-AHN_i
            HoogteVegetatie.save("Veg_hoogte")

            Slope_Vegetatie = Raster(arcpy.gp.Slope_sa(VegetatieAHN, "Veg_slope", 'DEGREE', '1'))
            print "AHN geanalyseerd"

            # Boom of niet
            Bomen = Con(HoogteVegetatie>2.0, 1) #1 is boom
            Bomen.save("Bomen")

            Vegetatieklasse = Con(HoogteVegetatie<1.0, 1, Con(HoogteVegetatie>2.0, 3, 2)) # 1 = gras, 2= struik, 3 = boom 
            #if hoogte > 2 meter dan boom
            Vegetatieklasse.save("Vegetatieklasse")

            # Vegetatieklasse = arcpy.RasterToPolygon_conversion(Klassen_ras, "Vegetatieklasse","SIMPLIFY", "Value")

            # arcpy.AddField_management(Vegetatieklasse,"Klasse_nr","TEXT",field_length=40)
            # cursor = arcpy.da.UpdateCursor(Vegetatieklasse, ["gridcode","Klasse_nr"])
            # for row in cursor:
            #     if row[0] == 1:#
            #         row[1] = "gras"
            #     elif row[0] == 2: #2
            #         row[1] = "Struik of zijkant boom"
            #     elif row[0] == 3: #3
            #         row[1] = "Boom"
            #     cursor.updateRow(row)

            ##################################################################################################################################
            Klassen = Con(Slope_Vegetatie<25,1,Con(HoogteVegetatie<1.5,2,Con(HoogteVegetatie>2,3,4))) #1= gras, 2=struik?, 3=boom, 4 =?
            Klassen.save("Klassen_gei")
            print "Klassen bepaald"
            # Explanation of Klassen cons 
            # f Slope_Vegetatie is < 25:
            #   return 1 (gras)
            # elif HoogteVegetatie < 1.5 meter:
            #   return 2 (struik?)
            # elif HoogteVegetatie > 2 meter:
            #   return 3 (boom?)
            # else: 
            #   return 4 (?????)

            ## arcpy.ApplySymbologyFromLayer_management("Klassen",r'C:\GEI\Projecten\KAS_vragen\CIR luchtfoto\Klassen.lyr')

        except:
            self.objTT.Traceback()
        finally:
            self.objTT.AddMessage("Klaar")
        return

if __name__ == '__main__':
    # This is used for debugging. Using this separated structure makes it much
    # easier to debug using standard Python development tools.

    tool = VegetatieScanner()
    params = tool.getParameterInfo()
    params[0].value = r'C:\GIS\Projecten\Nachthitte\Tilburg_test\ahn3.gdb\AHNI_Tilburg_test' #ahni
    params[1].value = r'C:\GIS\Projecten\Nachthitte\Tilburg_test\ahn3.gdb\AHNR_Tilburg_test' #ahnr
    params[2].value = r'C:\GIS\Projecten\Nachthitte\Tilburg_test\lufo.TIF' # lufo
    params[3].value = r'C:\GIS\Projecten\Nachthitte\Tilburg_test\VegScan.gdb #output_gdb' #output gdb

    tool.execute(parameters=params, messages=None)