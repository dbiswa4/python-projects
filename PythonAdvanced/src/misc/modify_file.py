import csv, os
import tempfile

def modify_file(file_name):
    # existingPDFFileName = 'PA_W2_R_Annual_Reconcilation_2012.pdf'
    existingPDFFileName = 'PA_W2-R_Annual_Reconciliation_2012.pdf'
    updatedExistingFormEndDate = '1000019'
    pdfFileNameColumnIndex = 4
    endDateColumnIndex = 20

    out = file('try_out.txt', 'w')

    with open(file_name, 'rb') as f:
        for row in f:
            fields = row.split("\t")
            #print 'field : ', fields[pdfFileNameColumnIndex]
            if fields[pdfFileNameColumnIndex].strip() == existingPDFFileName:
                #print 'found ', existingPDFFileName
                fields[endDateColumnIndex] = updatedExistingFormEndDate
                #print 'fields : ', fields

            out.write("\t".join(fields))
        out.close()


def modify_replace_file(file_name):
    # existingPDFFileName = 'PA_W2_R_Annual_Reconcilation_2012.pdf'
    existingPDFFileName = 'PA_W2-R_Annual_Reconciliation_2012.pdf'
    updatedExistingFormEndDate = '1000019'
    #updatedExistingFormEndDate = 'xxxxxxx'
    pdfFileNameColumnIndex = 4
    endDateColumnIndex = 20


    with tempfile.NamedTemporaryFile(dir='.', delete=False) as tmp, \
            open(file_name, 'rb') as f:
        for row in f:
            fields = row.split("\t")
            #print 'field : ', fields[pdfFileNameColumnIndex]
            if fields[pdfFileNameColumnIndex].strip() == existingPDFFileName:
                #print 'found ', existingPDFFileName
                fields[endDateColumnIndex] = updatedExistingFormEndDate
                #print 'fields : ', fields

            tmp.write("\t".join(fields))

    os.rename(tmp.name, file_name)


if __name__ == '__main__':
    modify_file('try.txt')
    modify_replace_file('try_replace.txt')
