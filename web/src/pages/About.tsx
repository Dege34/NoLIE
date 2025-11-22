import React from 'react'
import { ExternalLink, Shield, Eye, AlertTriangle, Github, BookOpen } from 'lucide-react'
import { t } from '@/lib/i18n'
import { useSettingsStore } from '@/store/settings'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export function About() {
  const { language } = useSettingsStore()

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold tracking-tight text-foreground mb-4">
            {t('about', language)}
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Understanding deepfake detection technology and its implications
          </p>
        </div>

        <div className="space-y-8">
          {/* What is Deepfake Forensics */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Eye className="h-5 w-5" />
                <span>What is NoLIE?</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                NoLIE is the field of detecting and analyzing manipulated media content, 
                particularly videos and images that have been created or altered using artificial intelligence. 
                Our system uses state-of-the-art machine learning models to identify subtle artifacts and 
                inconsistencies that indicate synthetic content.
              </p>
              <p className="text-muted-foreground">
                The technology combines multiple detection approaches including frequency domain analysis, 
                temporal consistency checks, and explainable AI methods to provide reliable and 
                interpretable results.
              </p>
            </CardContent>
          </Card>

          {/* How It Works */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Shield className="h-5 w-5" />
                <span>How It Works</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <span className="text-2xl font-bold text-primary">1</span>
                  </div>
                  <h3 className="font-semibold mb-2">Upload & Preprocess</h3>
                  <p className="text-sm text-muted-foreground">
                    Upload your media files and our system extracts frames, detects faces, 
                    and prepares the data for analysis.
                  </p>
                </div>
                
                <div className="text-center">
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <span className="text-2xl font-bold text-primary">2</span>
                  </div>
                  <h3 className="font-semibold mb-2">AI Analysis</h3>
                  <p className="text-sm text-muted-foreground">
                    Multiple neural networks analyze the content for deepfake indicators, 
                    including frequency artifacts and temporal inconsistencies.
                  </p>
                </div>
                
                <div className="text-center">
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-3">
                    <span className="text-2xl font-bold text-primary">3</span>
                  </div>
                  <h3 className="font-semibold mb-2">Results & Explanation</h3>
                  <p className="text-sm text-muted-foreground">
                    Get detailed results with confidence scores, heatmaps, and 
                    explanations to understand the detection process.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Ethics & Limitations */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <AlertTriangle className="h-5 w-5" />
                <span>Ethics & Limitations</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">Important Considerations</h4>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li>This tool is for research and educational purposes only</li>
                    <li>Results should not be used as the sole basis for important decisions</li>
                    <li>False positives and false negatives are possible</li>
                    <li>Model performance may vary across different types of content</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-2">Privacy & Security</h4>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li>Files are processed locally and not stored permanently</li>
                    <li>No personal data is collected or transmitted</li>
                    <li>All analysis is performed on your device when possible</li>
                    <li>Results are not shared with third parties</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-2">Limitations</h4>
                  <ul className="list-disc list-inside space-y-1 text-muted-foreground">
                    <li>Detection accuracy depends on the quality and type of content</li>
                    <li>Advanced deepfake techniques may be harder to detect</li>
                    <li>Results may be affected by compression and processing artifacts</li>
                    <li>Model performance may degrade over time as techniques evolve</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Technology Stack */}
          <Card>
            <CardHeader>
              <CardTitle>Technology Stack</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="font-semibold">PyTorch</div>
                  <div className="text-sm text-muted-foreground">Deep Learning</div>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="font-semibold">React</div>
                  <div className="text-sm text-muted-foreground">Frontend</div>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="font-semibold">FastAPI</div>
                  <div className="text-sm text-muted-foreground">Backend</div>
                </div>
                <div className="text-center p-4 bg-muted rounded-lg">
                  <div className="font-semibold">Docker</div>
                  <div className="text-sm text-muted-foreground">Deployment</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Resources */}
          <Card>
            <CardHeader>
              <CardTitle>Resources & Documentation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <a
                  href="https://github.com/deepfake-forensics/deepfake-forensics"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-accent transition-colors"
                >
                  <Github className="h-5 w-5" />
                  <div>
                    <div className="font-semibold">GitHub Repository</div>
                    <div className="text-sm text-muted-foreground">Source code and documentation</div>
                  </div>
                  <ExternalLink className="h-4 w-4 ml-auto" />
                </a>
                
                <a
                  href="https://deepfake-forensics.readthedocs.io"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center space-x-3 p-4 border rounded-lg hover:bg-accent transition-colors"
                >
                  <BookOpen className="h-5 w-5" />
                  <div>
                    <div className="font-semibold">Documentation</div>
                    <div className="text-sm text-muted-foreground">API reference and guides</div>
                  </div>
                  <ExternalLink className="h-4 w-4 ml-auto" />
                </a>
              </div>
            </CardContent>
          </Card>

          {/* Contact */}
          <Card>
            <CardHeader>
              <CardTitle>Contact & Support</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground mb-4">
                For questions, bug reports, or feature requests, please visit our GitHub repository 
                or contact us through the provided channels.
              </p>
              <div className="flex space-x-4">
                <Button variant="outline" asChild>
                  <a
                    href="https://github.com/deepfake-forensics/deepfake-forensics/issues"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Github className="h-4 w-4 mr-2" />
                    Report Issue
                  </a>
                </Button>
                <Button variant="outline" asChild>
                  <a
                    href="https://github.com/deepfake-forensics/deepfake-forensics/discussions"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <BookOpen className="h-4 w-4 mr-2" />
                    Discussions
                  </a>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
