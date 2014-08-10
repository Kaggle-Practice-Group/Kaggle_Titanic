library(shiny)
library(ggplot2)

tmp<- read.csv("data/train.csv", header=T, 
                colClasses=c("integer", "factor", "factor", "character", 
                             "factor", "numeric", "integer", "integer", 
                           "character", "numeric", "character", "factor"))
levels(tmp$Survived) = c("N", "Y")
levels(tmp$Sex) = c("F", "M")
train <<- tmp[, c(-1)]
filters <<- c("none",names(train))


shinyServer(function(input, output,session) {  
  
  data <- reactive({train})  
  
  output$colour <- renderUI({
    if (is.null(input$x))
      return()
    # Depending on input$x, we'll update the possible precitors.
    otherNames <- isolate({names(data())[!(names(data()) %in% c(input$y,input$x))]})
    selectInput("color", "Group",
                choices =  otherNames,
                selected = otherNames[[1]],width=150) 
  })

output$filter1 <- renderUI({
    if (is.null(input$f1))
      return()
  #   Depending on input$y, we'll update the possible precitors.
  items <- unique(data()[[input$f1]])  
  isolate(
    
    selectInput("fi1", "Filter Item",
                choices =  items,
                selected = items[1],multiple =T)) 
    
  })


output$filter2 <- renderUI({
  if (is.null(input$f2))
    return()
  #   Depending on input$y, we'll update the possible precitors.
  
  items <- unique(data()[[input$f2]])
  isolate(
  selectInput("fi2", "Filter Item",
              choices =  items,
              selected = items[1],multiple =T) )
})


output$filter3 <- renderUI({
  if (is.null(input$f3))
    return()
  #   Depending on input$y, we'll update the possible precitors.
  
  items <- unique(data()[[input$f3]])
  selectInput("fi3", "Filter Item",
              choices =  items,
              selected = items[1],multiple =T) 
})

output$filter4 <- renderUI({
  if (is.null(input$f4))
    return()
  #   Depending on input$y, we'll update the possible precitors.
  
  items <- unique(data()[[input$f4]])
  selectInput("fi4", "Filter Item",
              choices =  items,
              selected = items[1],multiple =T) 
})
  
  output$summary <- renderDataTable({
    if (is.null(input$y) || is.null(input$x))
      return()
    
    tmpdata = data()
    for (i in 1:4)
    {
      if (input[[paste('f', i, sep='')]]!='none')
      {
        switch(input[[paste('o', i, sep='')]],
               "=="={tmpdata <-  subset(tmpdata,tmpdata[[input[[paste('f', i, sep='')]]]]==input[[paste('fi', i, sep='')]])},
               "!="={tmpdata <-  subset(tmpdata,tmpdata[[input[[paste('f', i, sep='')]]]]!=input[[paste('fi', i, sep='')]])},
               ">="={tmpdata <-  subset(tmpdata,tmpdata[[input[[paste('f', i, sep='')]]]]>=input[[paste('fi', i, sep='')]])},
               "<="={tmpdata <-  subset(tmpdata,tmpdata[[input[[paste('f', i, sep='')]]]]<=input[[paste('fi', i, sep='')]])},
               ">"={tmpdata <-  subset(tmpdata,tmpdata[[input[[paste('f', i, sep='')]]]]>input[[paste('fi', i, sep='')]])},
               "<"={tmpdata <-  subset(tmpdata,tmpdata[[input[[paste('f', i, sep='')]]]]<input[[paste('fi', i, sep='')]])}
        )      
      }
    }
    if(input$fct=='none')
    {
      tmpdata
    }    
    else 
    {
      formula = as.formula(paste(input$x,input$y,sep="~"))    
      aggregate(formula,tmpdata,input$fct)  
    }    
  })
  
  
  output$plot <- renderPlot({
    if (is.null(input$y) || is.null(input$x) || is.null(input$color))
      return()
    
    tmpdata = data()
    tmpdata = data()
    for (i in 1:4)
    {
      if (input[[paste('f', i, sep='')]]!='none')
      {
        switch(input[[paste('o', i, sep='')]],
               "=="={tmpdata <-  subset(tmpdata,tmpdata[[input[[paste('f', i, sep='')]]]]==input[[paste('fi', i, sep='')]])},
               "!="={tmpdata <-  subset(tmpdata,tmpdata[[input[[paste('f', i, sep='')]]]]!=input[[paste('fi', i, sep='')]])},
               ">="={tmpdata <-  subset(tmpdata,tmpdata[[input[[paste('f', i, sep='')]]]]>=input[[paste('fi', i, sep='')]])},
               "<="={tmpdata <-  subset(tmpdata,tmpdata[[input[[paste('f', i, sep='')]]]]<=input[[paste('fi', i, sep='')]])},
               ">"={tmpdata <-  subset(tmpdata,tmpdata[[input[[paste('f', i, sep='')]]]]>input[[paste('fi', i, sep='')]])},
               "<"={tmpdata <-  subset(tmpdata,tmpdata[[input[[paste('f', i, sep='')]]]]<input[[paste('fi', i, sep='')]])}
        )      
      }
    }
    
    p<- ggplot(tmpdata, aes_string(x=input$x, y=input$y)) + 
      
      geom_point(aes_string(colour=input$color),size=3) 
    print(p)
  }, height=400)  
})