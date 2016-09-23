import sys

# create new GForm row and expire old row for any existing row whose pdf is "Local_Annual.pdf"; update the pdf and bind xml

#existingPDFFileName = 'PA_W2_R_Annual_Reconcilation_2012'
existingPDFFileName = 'PA_W2-R_Annual_Reconciliation_2012.pdf'
# newPDFFileName = 'LocalAnnual2013.pdf'
#
# newBindXMLName = 'Local_Annual2013.xml'

updatedExistingFormEndDate = '1000019'
# newEffectiveDate = '1/1/2013'


formIDColumnIndex = 0
pdfFileNameColumnIndex = 4
# effectiveDateColumnIndex = 9
endDateColumnIndex = 20
# bindXMLColumnIndex = 21

nextFormID = 0
glocaltaxforms = []
lineNumber = 0
#for line in open(sys.argv[1]).readlines():
for line in open("try.txt").readlines():
	columns = line.split("\t")
	glocaltaxforms.append(columns)
	lineNumber += 1
	if lineNumber > 1:
		nextFormID = max(nextFormID, int(columns[formIDColumnIndex]))

nextFormID += 1
newRows = []
for row in glocaltaxforms:
	pdfFileName = row[pdfFileNameColumnIndex]
	if pdfFileName.strip() == existingPDFFileName:
		# newRow = []
# 		newRows.append(newRow)
# 		newRow.extend(row)
# 		newRow[formIDColumnIndex] = str(nextFormID)
# 		nextFormID += 1
# 		newRow[pdfFileNameColumnIndex] = newPDFFileName
# 		newRow[effectiveDateColumnIndex] = newEffectiveDate
# 		newRow[bindXMLColumnIndex] = newBindXMLName
		row[endDateColumnIndex] = updatedExistingFormEndDate

#glocaltaxformsFile = open('./' + sys.argv[1] + '.updated.txt', 'w')
glocaltaxformsFile = open('./' + "try.txt" + '.updated.txt', 'w')
for row in glocaltaxforms:
	glocaltaxformsFile.write("\t".join(row))
# for newRow in newRows:
# 	glocaltaxformsFile.write("\t".join(newRow))
glocaltaxformsFile.close()