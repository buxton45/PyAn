Sub CreatePivotTable(group_by As String, addtnl_cols As String)
    Dim wsData As Worksheet
    Dim wsPivot As Worksheet
    Dim dataRange As Range
    Dim pivotCache As PivotCache
    Dim pt As PivotTable
    Dim pivotTableName As String
    Dim lastRow As Long
    Dim lastCol As Long
    Dim columnWidth As Double
    Dim groupArray() As String
    Dim additionalArray() As String
    Dim i As Long

    ' Set worksheet for data
    Set wsData = Worksheets("noIdxGroups") ' Change to your data sheet name
    
    ' Find the last row and column in the data
    lastRow = wsData.Cells(wsData.Rows.Count, "B").End(xlUp).Row ' Assumes data starts in column B
    lastCol = wsData.Cells(1, wsData.Columns.Count).End(xlToLeft).Column ' Assumes headers are in row 1

    ' Set the data range dynamically
    Set dataRange = wsData.Range(wsData.Cells(1, 2), wsData.Cells(lastRow, lastCol)) ' Adjusted to start from B1

    ' Add a new worksheet for the pivot table
    On Error Resume Next ' Ignore errors in case the sheet already exists
    Set wsPivot = Worksheets("PivotTableSheet")
    On Error GoTo 0 ' Turn error handling back on

    If wsPivot Is Nothing Then
        Set wsPivot = Worksheets.Add
        wsPivot.Name = "PivotTableSheet" ' Change name if needed
    Else
        wsPivot.Cells.Clear ' Clear the existing sheet if it already exists
    End If

    ' Define the pivot table name
    pivotTableName = "MyPivotTable"

    ' Create the pivot cache
    Set pivotCache = ThisWorkbook.PivotCaches.Create( _
        SourceType:=xlDatabase, _
        SourceData:=dataRange)

    ' Create the pivot table
    Set pt = pivotCache.CreatePivotTable( _
        TableDestination:=wsPivot.Cells(1, 1), _
        TableName:=pivotTableName)

    ' Configure the pivot table
    With pt
        ' Set up rows and columns
        ' Explicitly add "Date" and "Event Tag" as row fields
        .PivotFields("Date").Orientation = xlRowField
        .PivotFields("Event Tag").Orientation = xlRowField

        ' Set up additional rows from the group_by array
        groupArray = Split(group_by, ",")
        For i = LBound(groupArray) To UBound(groupArray)
            .PivotFields(groupArray(i)).Orientation = xlRowField
        Next i
        
        ' Set up values
        Dim fieldNames As Variant
        fieldNames = Array("PNs (#)", "PNs (%)", "Mom. Evs", "Am", "Bm", "Cm", "As", "Bs", "Cs", "A-", "B-", "C-", "Ax", "Bx", "Cx")
        
        Dim fName As Variant
        For Each fName In fieldNames
            With .PivotFields(fName)
                .Orientation = xlDataField
                ' Check if the field is "PNs (%)" to apply average, otherwise use sum
                If fName = "PNs (%)" Then
                    .Function = xlAverage ' Change to average for "PNs (%)"
                Else
                    .Function = xlSum ' Default to sum for other fields
                End If
                .NumberFormat = "#,##0"
            End With
        Next fName

        ' Split the addtnl_cols string into an array
        If Trim(addtnl_cols) <> "" Then
            additionalArray = Split(addtnl_cols, ",")
        Else
            additionalArray = Array() ' Empty array if addtnl_cols is empty
        End If

        ' Add additional fields to the pivot table
        For i = LBound(additionalArray) To UBound(additionalArray)
            If Not IsEmpty(additionalArray(i)) Then
                With .PivotFields(additionalArray(i))
                    .Orientation = xlDataField
                    If additionalArray(i) = "nPNs_w_gt13_mom" Then
                        .Function = xlSum
                        .NumberFormat = "#,##0"
                    Else
                        .Function = xlAverage
                        .NumberFormat = "#,##0.0000"
                    End If
                End With
            End If
        Next i
        
        ' Apply a pivot table style
        .TableStyle2 = "PivotStyleMedium9" ' Change to your desired style

        ' Sort by date first and then by "Sum of Mom. Evs"
        .PivotFields("Event Tag").AutoSort xlDescending, "Sum of Mom. Evs"
    End With

    ' Add borders to the Grand Total row
    Dim grandTotalRow As Range
    Dim dataEndColumn As Long
    dataEndColumn = pt.TableRange2.Columns.Count
    
    ' Get the range for the Grand Total row
    Set grandTotalRow = wsPivot.Range(wsPivot.Cells(pt.TableRange2.Rows.Count, 1), _
                                      wsPivot.Cells(pt.TableRange2.Rows.Count, dataEndColumn))

    ' Add a line above the Grand Total row
    With grandTotalRow.Borders(xlEdgeTop)
        .LineStyle = xlContinuous
        .Weight = xlThin
    End With

    ' Add a bold line below the Grand Total row
    With grandTotalRow.Borders(xlEdgeBottom)
        .LineStyle = xlContinuous
        .Weight = xlThick
    End With

    ' Call the subroutine to modify the PivotTable style
    Call ModifyPivotTableStyle(pt) ' Pass the PivotTable to modify styles

    ' Call the subroutine to apply color coding
    Call ApplyColorCoding(pt)

    ' Call the subroutine to rename pivot table fields
    Call RenamePivotTableFields(pt)

    ' Set all value columns to the same column width and center text
    columnWidth = 8 ' Set the desired column width here (adjust as needed)
    Dim dataField As PivotField
    Dim colIndex As Long
    For Each dataField In pt.DataFields
        colIndex = dataField.Position + 1 ' +1 because Columns is 1-based
        With wsPivot.Columns(colIndex)
            .ColumnWidth = columnWidth
            .HorizontalAlignment = xlCenter ' Center the text
        End With
    Next dataField

End Sub

Sub ApplyColorCoding(pt As PivotTable)
    Dim pf As PivotField
    Dim i As Long

    ' Color code the columns based on the specified criteria
    For i = 1 To pt.DataFields.Count
        Set pf = pt.DataFields(i)
        
        Select Case pf.Name
            Case "Sum of Am", "Sum of Bm", "Sum of Cm"
                pf.DataRange.Interior.Color = RGB(161, 201, 244) ' Light blue
            Case "Sum of As", "Sum of Bs", "Sum of Cs"
                pf.DataRange.Interior.Color = RGB(255, 180, 130) ' Light orange
            Case "Sum of A-", "Sum of B-", "Sum of C-"
                pf.DataRange.Interior.Color = RGB(141, 229, 161) ' Light green
            Case "Sum of Ax", "Sum of Bx", "Sum of Cx"
                pf.DataRange.Interior.Color = RGB(255, 159, 155) ' Light red
        End Select
    Next i
End Sub

Sub RenamePivotTableFields(pt As PivotTable)
    Dim pf As PivotField
    Dim newName As String

    ' Replace "YourSheetName" with the actual sheet name
    Set ws = Worksheets("PivotTableSheet")

    ' Replace "YourPivotTableName" with the actual pivot table name
    Set pt = ws.PivotTables("MyPivotTable")
    
    ' Rename data fields
    For Each pf In pt.DataFields
        newName = Replace(pf.Caption, "Sum of ", "")
        newName = Replace(newName, "Average of ", "") ' Remove "Average of " if needed
        newName = newName & " " ' Add space after, so name not repeat from original table
        pf.Caption = newName
    Next pf
End Sub

Sub ModifyPivotTableStyle(pt As PivotTable)
    Dim existingStyle As TableStyle
    Dim newStyle As TableStyle
    Dim styleName As String
    
    ' Get the existing style of the PivotTable
    Set existingStyle = ActiveWorkbook.TableStyles(pt.TableStyle2)
    
    ' Define a new style name
    styleName = "CustomStyleCopy"
    
    ' Check if the new style already exists, if not, create it
    On Error Resume Next
    Set newStyle = ActiveWorkbook.TableStyles(styleName)
    On Error GoTo 0
    
    If newStyle Is Nothing Then
        ' Create a new style
        Set newStyle = ActiveWorkbook.TableStyles.Add(styleName)
        
        ' Copy elements from the existing style to the new style
        With newStyle
            ' Example: Copy header row colors (add other elements as needed)
            .TableStyleElements(xlHeaderRow).Interior.Color      = existingStyle.TableStyleElements(xlHeaderRow).Interior.Color
            .TableStyleElements(xlRowSubheading1).Interior.Color = existingStyle.TableStyleElements(xlRowSubheading1).Interior.Color
            .TableStyleElements(xlRowSubheading2).Interior.Color = existingStyle.TableStyleElements(xlRowSubheading2).Interior.Color
        End With
    End If
    
    ' Modify the new style as needed
    With newStyle
        .TableStyleElements.Item(xlHeaderRow).Interior.Color = RGB(79, 129, 189) ' Set header color
        .TableStyleElements.Item(xlHeaderRow).Font.Bold = True ' Set font to bold for header
        .TableStyleElements.Item(xlHeaderRow).Font.Color = RGB(255, 255, 255) ' Set font to bold for header
        .TableStyleElements.Item(xlRowSubheading1).Interior.Color = RGB(141, 180, 226) ' Change color for Row Subheading 1
        .TableStyleElements.Item(xlRowSubheading1).Font.Bold = True ' Set font to bold for xlRowSubheading1
        .TableStyleElements.Item(xlRowSubheading2).Interior.Color = RGB(220, 230, 241) ' Change color for Row Subheading 2
        .TableStyleElements.Item(xlRowSubheading2).Font.Bold = True ' Set font to bold for xlRowSubheading2
        .TableStyleElements.Item(xlRowSubheading3).Font.Bold = True ' Set font to bold for xlRowSubheading3
    End With
    
    ' Apply the new style to the PivotTable
    pt.TableStyle2 = styleName
End Sub 

Sub RunAll(group_by As String, addtnl_cols As String)
    ' Run the CreatePivotTable routine
    Call CreatePivotTable(group_by, addtnl_cols)

    ' Remove the "noIdxGroups" worksheet
    ' On Error Resume Next ' Ignore errors if the sheet does not exist
    ' Application.DisplayAlerts = False ' Suppress any prompts
    ' Worksheets("noIdxGroups").Delete
    ' Application.DisplayAlerts = True ' Turn alerts back on
    ' On Error GoTo 0 ' Reset error handling
End Sub