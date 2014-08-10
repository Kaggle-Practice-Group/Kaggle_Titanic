library(shiny)
library(ggplot2)

stats <- c("lm","loess")

shinyUI(fluidPage(
  
  titlePanel("Simple data analysis"),
  
  fluidRow(
    column(4, wellPanel(
      selectInput('y', 'Response', names(train),names(train)[[1]]))),
    
    column(4, wellPanel(
      # Dynamic list of explanatory
      selectInput("x", "explanatory",
                  names(train),names(train)[[1]]))),
    
    column(4, wellPanel(
      # Grouping Colour
      uiOutput("colour")))  
    
    ),
  fluidRow(
    column(4, 
      wellPanel(                
        flowLayout(
          selectInput('f1', 'Column Filter',filters,filters[[1]],width=150) ,
          selectInput('o1', 'Operator',c("==","!=",">=","<=",">","<"),"==",width=70),
          uiOutput('filter1')),
                
        flowLayout(
          selectInput('f2', 'Column Filter',filters,filters[[1]],width=150) ,
          selectInput('o2', 'Operator',c("==","!=",">=","<=",">","<"),"==",width=70),
          uiOutput('filter2')),
        
        flowLayout(
          selectInput('f3', 'Column Filter',filters,filters[[1]],width=150) ,
          selectInput('o3', 'Operator',c("==","!=",">=","<=",">","<"),"==",width=70),
          uiOutput('filter3')),
        flowLayout(
          selectInput('f4', 'Column Filter',filters,filters[[1]],width=150) ,
          selectInput('o4', 'Operator',c("==","!=",">=","<=",">","<"),"==",width=70),
          uiOutput('filter4')))
        
      ),

column(8,
       tabsetPanel(
          tabPanel("Summary",wellPanel(
            selectInput('fct', 'Function',c("none","sum","mean","sd","max","min","median","length"),"length"),
                   dataTableOutput('summary'))),
          tabPanel("Plot", plotOutput('plot'))
       )

    ))
  
))